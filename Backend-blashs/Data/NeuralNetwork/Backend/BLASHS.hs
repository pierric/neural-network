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
{-# LANGUAGE UndecidableInstances #-}
module Data.NeuralNetwork.Backend.BLASHS (
  -- module Data.NeuralNetwork.Backend.BLASHS.Layers,
  module Data.NeuralNetwork.Backend.BLASHS.Utils,
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
import Data.NeuralNetwork.Backend.BLASHS.SIMD
import Control.Monad.Except (throwError)
import Control.Monad.State
import Data.Constraint (Dict(..))
import Blas.Generic.Unsafe (Numeric)

-- | The backend data type
data ByBLASHS p = ByBLASHS

byBLASHSf :: ByBLASHS Float
byBLASHSf = ByBLASHS
byBLASHSd :: ByBLASHS Double
byBLASHSd = ByBLASHS

type AbbrSpecToCom p s   = SpecToCom (ByBLASHS p) s
type AbbrSpecToEvl p s o = SpecToEvl (ByBLASHS p) (AbbrSpecToCom p s) o

-- | Neural network specified to start with 1D / 2D input
instance (InputLayer i, RealType p,
          BodyTrans  (ByBLASHS p) s,
          EvalTrans  (ByBLASHS p) (AbbrSpecToCom p s) o,
          ModelCst (AbbrSpecToCom p s) (AbbrSpecToEvl p s o))
       => Backend (ByBLASHS p) (i,s,o) where
  type ComponentFromSpec (ByBLASHS p) (i,s,o) = AbbrSpecToCom p s
  type EvaluatorFromSpec (ByBLASHS p) (i,s,o) = AbbrSpecToEvl p s o
  compile b (i,s,o) = do c <- btrans b (isize i) s
                         e <- etrans b c o
                         return $ (c, e)
  witness _ _ = Dict

instance (Numeric p, RealType p, SIMDable p) => BodyTrans (ByBLASHS p) SpecFullConnect where
  -- 'SpecFullConnect' is translated to a two-layer component
  -- a full-connect, followed by a relu activation (1D, single channel)
  type SpecToCom (ByBLASHS p) SpecFullConnect = Stack (RunLayer p F) (RunLayer p (T SinglVec))
  btrans _ (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                       return $ Stack u (Activation (relu, relu'))
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans (ByBLASHS p) SpecConvolution where
  -- 'SpecConvolution' is translated to a two-layer component
  -- a convolution, following by a relu activation (2D, multiple channels)
  type SpecToCom (ByBLASHS p) SpecConvolution = Stack (RunLayer p C) (RunLayer p (T MultiMat))
  btrans _ (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p
                                               return $ Stack u (Activation (relu, relu'))
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans (ByBLASHS p) SpecMaxPooling where
  -- 'MaxPooling' is translated to a max-pooling component.
  type SpecToCom (ByBLASHS p) SpecMaxPooling = RunLayer p P
  btrans _ (D2 _ _ _) (MaxPooling n) = return (MaxP n)
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans (ByBLASHS p) SpecReshape2DAs1D where
  -- 'SpecReshape2DAs1D' is translated to a reshaping component.
  type SpecToCom (ByBLASHS p) SpecReshape2DAs1D = Reshape2DAs1D p
  btrans _ (D2 _ _ _) _ = return as1D
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans (ByBLASHS p) SpecLSTM where
  -- 'SpecLSTM' is translated to a LSTM component.
  type SpecToCom (ByBLASHS p) SpecLSTM = Stack (LSTM p) (RunLayer p (T SinglVec))
  btrans _ (D1 s) (LSTM n) = do u <- lift $ newLSTM s n
                                return $ Stack u (Activation (relu, relu'))
  btrans _ _ _ = throwError ErrMismatch

instance (BodyTrans (ByBLASHS p) a) => BodyTrans (ByBLASHS p) (SpecFlow a) where
  --
  type SpecToCom (ByBLASHS p) (SpecFlow a) = Stream (SpecToCom (ByBLASHS p) a)
  btrans b (SV s) (Flow a) = do u <- btrans b s a
                                return $ Stream u
  btrans _ _ _ = throwError ErrMismatch

instance Component c => EvalTrans (ByBLASHS p) c SpecEvaluator where
  type SpecToEvl (ByBLASHS p) c SpecEvaluator = Eval p
  etrans _ _ = return . Eval
