------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module defines an abstract interface for neural network
-- and a protocol for its backends to follow.
------------------------------------------------------------
module Data.NeuralNetwork (
  module Data.NeuralNetwork.Model,
  module Data.NeuralNetwork.Common
) where

import Data.NeuralNetwork.Model
import Data.NeuralNetwork.Common

