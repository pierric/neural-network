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
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE Rank2Types #-}
module Data.NeuralNetwork.Stack (
  Stack(..),
  StackLift(..)
) where

import Data.Data
import Data.NeuralNetwork

data Stack a b = Stack a b
  deriving (Typeable, Data)
data StackLift a b w = StackLift a b (forall t. Run a t -> w t) (forall t. Run b t -> w t)
  deriving (Typeable)

instance (Data a, Data b, Typeable w) => Data (StackLift a b w) where
  toConstr _ = stackLiftConstr
  gfoldl f z (StackLift a b u v) = z (\a b -> StackLift a b u v) `f` a `f` b
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(StackLift)"
  dataTypeOf _  = stackLiftDataType

stackLiftConstr   :: Constr
stackLiftConstr = mkConstr stackLiftDataType "StackLift" ["component1", "component2"] Prefix
stackLiftDataType :: DataType
stackLiftDataType = mkDataType "Data.NeuralNetwork.Stack" [stackLiftConstr]

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
          Monad (Run a), Monad (Run b), Monad w, Typeable w,
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
