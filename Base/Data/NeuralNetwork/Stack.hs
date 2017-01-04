------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Stack
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module defines an general mechanism to stack two
-- compatible neural network component.
------------------------------------------------------------
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE Rank2Types #-}
module Data.NeuralNetwork.Stack (
  Stack(..), StackLift(..)
) where

import Data.NeuralNetwork

data Stack a b = Stack a b
data StackLift a b w = StackLift a b (forall t. Run a t -> w t) (forall t. Run b t -> w t)

instance (Component a, Component b,
          Run a ~ Run b, Monad (Run a),
          Out a ~ Inp b
         ) => Component (Stack a b) where
  type Run (Stack a b) = Run a
  type Inp (Stack a b) = Inp a
  type Out (Stack a b) = Out b
  newtype Trace (Stack a b) = S0Trace (Trace b, Trace a)
  forwardT (Stack a b) !i = do
    !tra <- forwardT a i
    !trb <- forwardT b (output tra)
    return $ S0Trace (trb, tra)
  output (S0Trace !a) = output (fst a)
  backward (Stack a b) (S0Trace (!trb,!tra)) !odeltb rate = do
    (b', !odelta) <- backward b trb odeltb rate
    (a', !idelta) <- backward a tra odelta rate
    return (Stack a' b', idelta)

instance (Component a, Component b,
          Monad (Run a), Monad (Run b), Monad w,
          Out a ~ Inp b
         ) => Component (StackLift a b w) where
  type Run (StackLift a b w) = w
  type Inp (StackLift a b w) = Inp a
  type Out (StackLift a b w) = Out b
  newtype Trace (StackLift a b w) = S1Trace (Trace b, Trace a)
  forwardT (StackLift a b la lb) !i = do
    !tra <- la $ forwardT a i
    !trb <- lb $ forwardT b (output tra)
    return $ S1Trace (trb, tra)
  output (S1Trace !a) = output (fst a)
  backward (StackLift a b la lb) (S1Trace (!trb,!tra)) !odeltb rate = do
    (b', !odelta) <- lb $ backward b trb odeltb rate
    (a', !idelta) <- la $ backward a tra odelta rate
    return (StackLift a' b' la lb, idelta)
