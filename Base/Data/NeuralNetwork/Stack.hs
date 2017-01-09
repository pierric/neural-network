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
module Data.NeuralNetwork.Stack (
  Stack(..),
  CE, CL, CR, LiftRun
) where

import Data.Data
import Control.Monad.Trans
import Data.NeuralNetwork

data Stack a b c = Stack a b
  deriving (Typeable, Data)

data CE
data CL (t :: (* -> *) -> * -> *)
data CR (t :: (* -> *) -> * -> *)
type family LiftRun (u :: * -> *) (v :: * -> *) where
  LiftRun (t m) m = CL t
  LiftRun m (t m) = CR t
  LiftRun m m     = CE

instance (Component a, Component b,
          Monad (Run a), Monad (Run b),
          Run a ~ Run b,
          Out a ~ Inp b
         ) => Component (Stack a b CE) where
  type Run (Stack a b CE) = Run a
  type Inp (Stack a b CE) = Inp a
  type Out (Stack a b CE) = Out b
  newtype Trace (Stack a b CE) = S0Trace (Trace b, Trace a)
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
          Monad (Run a), Monad (Run b),
          MonadTrans t, Run a ~ t (Run b),
          Out a ~ Inp b
         ) => Component (Stack a b (CL t)) where
  type Run (Stack a b (CL t)) = Run a
  type Inp (Stack a b (CL t)) = Inp a
  type Out (Stack a b (CL t)) = Out b
  newtype Trace (Stack a b (CL t)) = S1Trace (Trace b, Trace a)
  forwardT (Stack a b) !i = do
    !tra <- forwardT a i
    !trb <- lift $ forwardT b (output tra)
    return $ S1Trace (trb, tra)
  output (S1Trace !a) = output (fst a)
  backward (Stack a b) (S1Trace (!trb,!tra)) !odeltb rate = do
    (b', !odelta) <- lift $ backward b trb odeltb rate
    (a', !idelta) <- backward a tra odelta rate
    return (Stack a' b', idelta)

instance (Component a, Component b,
          Monad (Run a), Monad (Run b),
          MonadTrans t, Run b ~ t (Run a),
          Out a ~ Inp b
         ) => Component (Stack a b (CR t)) where
  type Run (Stack a b (CR t)) = Run b
  type Inp (Stack a b (CR t)) = Inp a
  type Out (Stack a b (CR t)) = Out b
  newtype Trace (Stack a b (CR t)) = S2Trace (Trace b, Trace a)
  forwardT (Stack a b) !i = do
    !tra <- lift $ forwardT a i
    !trb <- forwardT b (output tra)
    return $ S2Trace (trb, tra)
  output (S2Trace !a) = output (fst a)
  backward (Stack a b) (S2Trace (!trb,!tra)) !odeltb rate = do
    (b', !odelta) <- backward b trb odeltb rate
    (a', !idelta) <- lift $ backward a tra odelta rate
    return (Stack a' b', idelta)
