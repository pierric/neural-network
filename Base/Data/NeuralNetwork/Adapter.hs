------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Adapter
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module defines an general adapter component.
------------------------------------------------------------
module Data.NeuralNetwork.Adapter (
  Adapter(..)
) where

import Data.NeuralNetwork.Model

data Adapter m i o e = Adapter (i -> m (e,o)) (e -> o -> m i)

instance (RunInIO m) => Component (Adapter m i o e) where
  type Run (Adapter m i o e) = m
  type Inp (Adapter m i o e) = i
  type Out (Adapter m i o e) = o
  newtype Trace (Adapter m i o e) = ATrace (e,o)
  forwardT (Adapter f _) i = ATrace <$> (f i)
  output (ATrace (_,o)) = o
  backward a@(Adapter _ b) (ATrace (e,_)) o _ = b e o >>= \i -> return (a,i)
