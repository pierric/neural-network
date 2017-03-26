------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS.Optimizer
-- Description :  A backend for neural network on top of 'blas-hs'
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module supplies a simple optimizer.
------------------------------------------------------------
module Data.NeuralNetwork.Backend.BLASHS.Optimizer (
  BlasOptimizer
) where

import Data.Data
import GHC.Float (float2Double)
import Data.NeuralNetwork
import Data.NeuralNetwork.Backend.BLASHS.Utils

newtype BlasOptimizer = BlasOptimizer SpecOptimizer
  deriving (Typeable, Data)

class BlasOptimizable a where
  scale :: Float -> a -> IO a
instance BlasOptimizable (Scalar Float) where
  scale rate sca = return $ Scalar (unScalar sca * rate)
instance BlasOptimizable (Scalar Double) where
  scale rate sca = return $ Scalar (unScalar sca * float2Double rate)
instance BlasOptimizable (DenseVector Float) where
  scale rate vec = do vec <<= Scale rate
                      return vec
instance BlasOptimizable (DenseVector Double) where
  scale rate vec = do vec <<= Scale (float2Double rate)
                      return vec
instance BlasOptimizable (DenseMatrix Float) where
  scale rate mat = do mat <<= Scale rate
                      return mat
instance BlasOptimizable (DenseMatrix Double) where
  scale rate mat = do mat <<= Scale (float2Double rate)
                      return mat
instance BlasOptimizable (DenseMatrixArray Float) where
  scale rate mta = do mta <<= Scale rate
                      return mta
instance BlasOptimizable (DenseMatrixArray Double) where
  scale rate mta = do mta <<= Scale (float2Double rate)
                      return mta

instance Optimizer BlasOptimizer where
  data OptVar BlasOptimizer b = OptVar SpecOptimizer
  type OptCst BlasOptimizer   = BlasOptimizable
  newOptVar (BlasOptimizer meth@(ScaleGrad _)) _ = return $ OptVar meth
  newOptVar _ _ = error ""
  optimize (OptVar (ScaleGrad rate)) v = scale rate v
  optimize _ _ = error ""
