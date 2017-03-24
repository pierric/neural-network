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
  learn,
  module Data.NeuralNetwork.Common
) where

import Control.Monad.Trans (liftIO)
import Data.Data
import Data.NeuralNetwork.Common

-- | By giving a way to measure the error, 'learn' can update the
-- neural network component.
learn :: (ModelCst n e, Optimizer o, Data (n o))
      => (n o,e)                            -- ^ neuron network
      -> (Inp n, Val e)                     -- ^ input and expect output
      -> Run n (n o,e)                      -- ^ updated network
learn (n,e) (i,o) = do
    tr <- forwardT n i
    o' <- liftIO $ eval e (output tr)
    er <- liftIO $ cost e o' o
    n' <- fst <$> backward n tr er
    return (n', e)
