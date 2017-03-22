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
import Control.Monad.Except (ExceptT, throwError)
import Control.Monad.State
import Data.Constraint (Dict(..))
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

type AbbrSpecToCom p s o   = SpecToCom (ByBLASHS p) o s
type AbbrSpecToEvl p s o v = SpecToEvl (ByBLASHS p) (AbbrSpecToCom p o s) v

-- | Neural network specified to start with 1D / 2D input
instance (InputLayer i, RealType p,
          BodyTrans  Err (ByBLASHS p) s,
          EvalTrans  Err (ByBLASHS p) (AbbrSpecToCom p s) o,
          BackendCst Err (AbbrSpecToCom p s) (AbbrSpecToEvl p s o))
       => Backend (ByBLASHS p) (i,s,o) where
  type Env (ByBLASHS p) = Err
  type ComponentFromSpec (ByBLASHS p) (i,s,o) = AbbrSpecToCom p s
  type EvaluatorFromSpec (ByBLASHS p) (i,s,o) = AbbrSpecToEvl p s o
  compile b (i,s,o) = do c <- btrans b (isize i) s
                         e <- etrans b c o
                         return $ (c, e)
  witness _ _ = Dict

instance RunInEnv IO Err where
  run = liftIO

instance (Numeric p, RealType p, SIMDable p) => BodyTrans Err (ByBLASHS p) SpecFullConnect where
  -- 'SpecFullConnect' is translated to a two-layer component
  -- a full-connect, followed by a relu activation (1D, single channel)
  type SpecToCom (ByBLASHS p) o SpecFullConnect = Stack (FullConn o p) (ActivateS p) CE
  btrans _ (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                       return $ Stack u (ActivateS (relu, relu'))
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans Err (ByBLASHS p) SpecConvolution where
  -- 'SpecConvolution' is translated to a two-layer component
  -- a convolution, following by a relu activation (2D, multiple channels)
  type SpecToCom (ByBLASHS p) o SpecConvolution = Stack (Convolute o p) (ActivateM p) CE
  btrans _ (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p
                                               return $ Stack u (ActivateM (relu, relu'))
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans Err (ByBLASHS p) SpecMaxPooling where
  -- 'MaxPooling' is translated to a max-pooling component.
  type SpecToCom (ByBLASHS p) o SpecMaxPooling = MaxPool p
  btrans _ (D2 _ _ _) (MaxPooling n) = return (MaxPool n)
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans Err (ByBLASHS p) SpecReshape2DAs1D where
  -- 'SpecReshape2DAs1D' is translated to a reshaping component.
  type SpecToCom (ByBLASHS p) SpecReshape2DAs1D = Reshape2DAs1D p
  btrans _ (D2 _ _ _) _ = return as1D
  btrans _ _ _ = throwError ErrMismatch

instance (Numeric p, RealType p, SIMDable p) => BodyTrans Err (ByBLASHS p) SpecLSTM where
  -- 'SpecLSTM' is translated to a LSTM component.
  type SpecToCom (ByBLASHS p) o SpecLSTM = Stack (LSTM o p) (ActivateS p) (LiftRun (Run (LSTM o p)) (Run (ActivateS p)))
  btrans _ (D1 s) (LSTM n) = do u <- lift $ newLSTM s n
                                return $ Stack u (ActivateS (relu, relu'))
  btrans _ _ _ = throwError ErrMismatch

instance (BodyTrans Err (ByBLASHS p) a) => BodyTrans Err (ByBLASHS p) (SpecFlow a) where
  --
  type SpecToCom (ByBLASHS p) (SpecFlow a) = Stream (SpecToCom (ByBLASHS p) a)
  btrans b (SV s) (Flow a) = do u <- btrans b s a
                                return $ Stream u
  btrans _ _ _ = throwError ErrMismatch

instance Component c => EvalTrans Err (ByBLASHS p) c SpecEvaluator where
  type SpecToEvl (ByBLASHS p) c SpecEvaluator = Eval (Run c) p
  etrans _ _ = return . Eval
