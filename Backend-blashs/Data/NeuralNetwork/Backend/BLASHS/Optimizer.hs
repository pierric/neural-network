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
) where

instance Optimizer SpecOptimizer where
  definitions
