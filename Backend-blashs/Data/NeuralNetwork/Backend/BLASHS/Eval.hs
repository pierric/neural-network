------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS.Eval
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
module Data.NeuralNetwork.Backend.BLASHS.Eval (
  Eval(..)
) where

import Data.NeuralNetwork
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD
import Control.Monad.Trans (MonadIO)

data Eval (m :: * -> *) = Eval SpecEvaluator

mse_eval x   = return x
mse_cost x y = do v <- newDenseVector (size x)
                  v <<= ZipWith cost' x y
                  return v

instance MonadIO m => Evaluator m (Eval m) where
  type Val (Eval m) = DenseVector Float
  eval (Eval MeanSquaredError)    = mse_eval
  eval (Eval SoftmaxCrossEntropy) = error ""
  cost (Eval MeanSquaredError)    = mse_cost
  cost (Eval SoftmaxCrossEntropy) = error ""
