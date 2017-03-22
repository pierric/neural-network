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
{-# LANGUAGE ConstraintKinds #-}
module Data.NeuralNetwork (
  Component(..),
  Evaluator(..),
  learn,
  ModelCst, BackendCst,
  Backend(..),
  RunInEnv(..),
  module Data.NeuralNetwork.Common
) where

import Data.Constraint
import Data.NeuralNetwork.Common

-- | Abstraction of a neural network component
class Component a where
  type Dty a :: *
  -- | execution environment
  type Run a :: * -> *
  -- | the type of input and in-error
  type Inp a
  -- | the type of output and out-error
  type Out a
  -- | the trace of a forward propagation
  data Trace a
  -- | Forward propagation
  forwardT :: a -> Inp a -> Run a (Trace a)
  -- | Forward propagation
  forward  :: Applicative (Run a) => a -> Inp a -> Run a (Out a)
  forward a = (output <$>) . forwardT a
  -- | extract the output value from the trace
  output   :: Trace a -> Out a
  -- | Backward propagation
  backward :: a -> Trace a -> Out a -> Run a (a, Inp a)

class Monad m => Evaluator m a where
  type Val a
  eval :: a -> Val a -> m (Val a)
  cost :: a -> Val a -> Val a -> m (Val a)

type ModelCst   a b   = (Component a, Monad (Run a), Evaluator (Run a) b, Out a ~ Val b)
type BackendCst e a b = (ModelCst a b, RunInEnv (Run a) e)

-- | By giving a way to measure the error, 'learn' can update the
-- neural network component.
learn :: (ModelCst n e)
      => (n,e)                              -- ^ neuron network
      -> (Inp n, Val e)                     -- ^ input and expect output
      -> Run n (n,e)                        -- ^ updated network
learn (n,e) (i,o) = do
    tr <- forwardT n i
    o' <- eval e (output tr)
    er <- cost e o' o
    n' <- fst <$> backward n tr er
    return (n', e)

-- | Abstraction of backend to carry out the specification
class Backend b s where
  -- | environment to 'compile' the specification
  type Env b :: * -> *
  -- | result type of 'compile'
  type ComponentFromSpec b s :: *
  type EvaluatorFromSpec b s :: *
  -- | necessary constraints of the resulting type
  witness :: b -> s -> Dict ( Monad (Env b)
                            , BackendCst (Env b) (ComponentFromSpec b s) (EvaluatorFromSpec b s))
  -- | compile the specification to runnable component.
  compile :: Optimizer o => b -> s -> o -> Env b ((ComponentFromSpec b s), (EvaluatorFromSpec b s))

-- | Lifting from one monad to another.
-- It is not necessary that the 'Env' and 'Run' maps to the
-- same execution environment, but the 'Run' one should be
-- able to be lifted to 'Env' one.
class (Monad r, Monad e) => RunInEnv r e where
  run :: r a -> e a
