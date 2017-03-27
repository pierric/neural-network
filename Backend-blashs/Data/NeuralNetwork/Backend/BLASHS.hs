------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS
-- Description :  A backend for neural network on top of 'blas-hs'
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module supplies a backend for the neural-network-base
-- package. This backend is implemented on top of the blas-hs
-- package and optimised with SIMD.
------------------------------------------------------------
{-# LANGUAGE UndecidableInstances, DataKinds #-}
module Data.NeuralNetwork.Backend.BLASHS (
  -- module Data.NeuralNetwork.Backend.BLASHS.Layers,
  module Data.NeuralNetwork.Backend.BLASHS.Utils,
  module Data.NeuralNetwork.Backend.BLASHS.Optimizer,
  module Data.NeuralNetwork.Backend.BLASHS.LSTM,
  module Data.NeuralNetwork.Backend.BLASHS.SIMD,
  ByBLASHS(..), byBLASHSf, byBLASHSd
) where

import Data.NeuralNetwork
import Data.NeuralNetwork.Stack
import Data.NeuralNetwork.Common
import Data.NeuralNetwork.Backend.BLASHS.Layers
import Data.NeuralNetwork.Backend.BLASHS.LSTM
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.Eval
import Data.NeuralNetwork.Backend.BLASHS.Optimizer
import Data.NeuralNetwork.Backend.BLASHS.SIMD
import Control.Monad.Except (ExceptT, throwError)
import Control.Monad.State
import Data.Constraint (Dict(..), withDict)
import Blas.Generic.Unsafe (Numeric)

-- | Compilation of the specification of a neural network is carried out in
-- the 'Err' monad, and the possible errors are characterized by 'ErrCode'.
type Err     = ExceptT ErrCode IO

-- | The backend data type
data ByBLASHS p = ByBLASHS

byBLASHSf :: ByBLASHS Float
byBLASHSf = ByBLASHS
byBLASHSd :: ByBLASHS Double
byBLASHSd = ByBLASHS

type ByBLASHSCompilable p s = (Precision p, BodyTrans (ByBLASHS p) s,
                               Out (SpecToCom (ByBLASHS p) s) ~ DenseVector p,
                               MonadIO (Run (SpecToCom (ByBLASHS p) s)))

-- | Neural network specified to start with 1D / 2D input
instance Backend (ByBLASHS p) where
  type Env (ByBLASHS p) = Err
  type CompileComponent (ByBLASHS p) s = SpecToCom (ByBLASHS p) s
  type CompileEvaluator (ByBLASHS p)   = Eval p
  type CompileOptimizer (ByBLASHS p)   = BlasOptimizer
  type Optimizable (ByBLASHS p) opt = (OptCst opt (Scalar p), OptCst opt (DenseVector p), OptCst opt (DenseMatrix p), OptCst opt (DenseMatrixArray p))
  type Compilable  (ByBLASHS p) s   = (ByBLASHSCompilable p s)
  compile b opt e i s = do c <- btrans b opt (isize i) s
                           withDict (bwitness b opt s) $ return (c, Eval e)
  mkopt b s = return $ BlasOptimizer s
  witness b opt e s = withDict (bwitness b opt s) Dict

instance Precision p => BodyTrans (ByBLASHS p) SpecFullConnect where
  -- 'SpecFullConnect' is translated to a two-layer component
  -- a full-connect, followed by a relu activation (1D, single channel)
  type SpecToCom (ByBLASHS p) SpecFullConnect = Stack (FullConn p) (ActivateS p)
  btrans _ o (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n o
                                         return $ Stack u (ActivateS relu relu') (Dict, Dict)
  btrans _ _ _ _ = throwError ErrMismatch
  bwitness _ _ _ = Dict

instance Precision p => BodyTrans (ByBLASHS p) SpecConvolution where
  -- 'SpecConvolution' is translated to a two-layer component
  -- a convolution, following by a relu activation (2D, multiple channels)
  type SpecToCom (ByBLASHS p) SpecConvolution = Stack (Convolute p) (ActivateM p)
  btrans _ o (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p o
                                                 return $ Stack u (ActivateM relu relu') (Dict, Dict)
  btrans _ _ _ _ = throwError ErrMismatch
  bwitness _ _ _ = Dict

instance Precision p => BodyTrans (ByBLASHS p) SpecMaxPooling where
  -- 'MaxPooling' is translated to a max-pooling component.
  type SpecToCom (ByBLASHS p) SpecMaxPooling = MaxPool p
  btrans _ _ (D2 _ _ _) (MaxPooling n) = return (MaxPool n)
  btrans _ _ _ _ = throwError ErrMismatch
  bwitness _ _ _ = Dict

instance Precision p => BodyTrans (ByBLASHS p) SpecReshape2DAs1D where
  -- 'SpecReshape2DAs1D' is translated to a reshaping component.
  type SpecToCom (ByBLASHS p) SpecReshape2DAs1D = Reshape2DAs1D p
  btrans _ _ (D2 _ _ _) _ = return as1D
  btrans _ _ _ _= throwError ErrMismatch
  bwitness _ _ _ = Dict

instance Precision p => BodyTrans (ByBLASHS p) SpecLSTM where
  -- 'SpecLSTM' is translated to a LSTM component.
  type SpecToCom (ByBLASHS p) SpecLSTM = Stack (LSTM p) (Lift (LSTM_Env_Transformer p) (ActivateS p))
  btrans _ o (D1 s) (LSTM n) = do u <- lift $ newLSTM s n o
                                  return $ Stack u (Lift (ActivateS relu relu') Dict) (Dict, Dict)
  btrans _ _ _ _ = throwError ErrMismatch
  bwitness _ _ _ = Dict

type Streamable p a = (p ~ Dty (SpecToCom (ByBLASHS p) a),
                       Inp (SpecToCom (ByBLASHS p) a) ~ DenseVector p,
                       Run (SpecToCom (ByBLASHS p) a) ~ LSTM_Env_Transformer p IO)
instance (Precision p, BodyTrans (ByBLASHS p) a, Streamable p a) =>
  BodyTrans (ByBLASHS p) (SpecFlow a) where
  --
  type SpecToCom (ByBLASHS p) (SpecFlow a) = Stream (SpecToCom (ByBLASHS p) a)
  btrans b o (SV s) (Flow a) = do u <- btrans b o s a
                                  withDict (bwitness b o a) $
                                    return (Stream u Dict)
  btrans _ _ _ _ = throwError ErrMismatch
  bwitness b o (Flow a) = withDict (bwitness b o a) $ Dict
