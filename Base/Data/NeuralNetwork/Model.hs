------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Model
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2017 Jiasen Wu
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
module Data.NeuralNetwork.Model (
  Component(..),
  Evaluator(..),
  Dty,
  learn,
  ModelCst,
  Backend(..),
  RunInIO(..),
  module Data.NeuralNetwork.Common
) where

import Data.Constraint
import Data.NeuralNetwork.Common

-- | Abstraction of a neural network component
class (RunInIO (Run a)) => Component a where
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
  backward :: a -> Trace a -> Out a -> Float -> Run a (a, Inp a)

class Evaluator a where
  type Val a
  eval :: a -> Val a -> IO (Val a)
  cost :: a -> Val a -> Val a -> IO (Val a)

-- | A model is a pair of some component and some evaluator
type Model    c e = (c, e)
-- | Model's type constraint
type ModelCst c e = (Component c, Evaluator e, Out c ~ Val e)

-- | Get the data type (float/double) from the model
type family Dty m :: *


-- | By giving a way to measure the error, 'learn' can update the
-- neural network component.
learn :: (ModelCst n e)
      => Model n e                          -- ^ neuron network
      -> (Inp n, Val e)                     -- ^ input and expect output
      -> Float                              -- ^ learning rate
      -> IO (Model n e)                     -- ^ updated network
learn (n,e) (i,o) rate = do
  tr <- run $ forwardT n i
  o' <- eval e (output tr)
  er <- cost e o' o
  n' <- run $ (fst <$> backward n tr er rate)
  return (n', e)

-- | Abstract backend, which should define how to translate a specification
-- into some component and some evaluator.
class Backend back spec where
  -- | result type of 'compile'
  type ComponentFromSpec back spec :: *
  type EvaluatorFromSpec back spec :: *
  -- | necessary constraints of the resulting type
  witness :: back -> spec -> Dict (ModelCst (ComponentFromSpec back spec) (EvaluatorFromSpec back spec))
  -- | compile the specification to runnable component.
  compile :: back -> spec -> Err (Model (ComponentFromSpec back spec) (EvaluatorFromSpec back spec))

-- | Lifting from one monad to another.
-- It is not necessary that the 'Env' and 'Run' maps to the
-- same execution environment, but the 'Run' one should be
-- able to be lifted to 'Env' one.
class Monad r => RunInIO r where
  run :: r a -> IO a

instance RunInIO IO where
  run = id