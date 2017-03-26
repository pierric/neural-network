------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Common
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module defines common interfaces for backends
------------------------------------------------------------
{-# LANGUAGE DataKinds, TypeOperators #-}
{-# LANGUAGE ConstraintKinds          #-}
module Data.NeuralNetwork.Common(
  RealType(..),
  LayerSize(..),
  InputLayer(..),
  BodySize(..), BodyTrans(..),
  Component(..),
  Evaluator(..),
  ErrCode(..),
  Backend(..),
  ModelCst,
  HVect((:&:),HNil),
  Optimizer(..),
) where

import Control.Monad.Except (MonadError)
import Control.Monad.Trans (MonadIO)
import GHC.Float (double2Float, float2Double)
import Data.Constraint (Dict(..))
import GHC.Exts (Constraint)
import Data.Data
import Data.HVect

class Data a => Optimizer a where
  data OptVar a b
  type OptCst a :: * -> Constraint
  newOptVar :: OptCst a b => a -> b -> IO (OptVar a b)
  optimize  :: OptCst a b => OptVar a b -> b -> IO b

class (Fractional a, Ord a) => RealType a where
  fromDouble :: Double -> a
  fromFloat  :: Float  -> a

instance RealType Float where
  fromDouble = double2Float
  fromFloat  = id

instance RealType Double where
  fromDouble = id
  fromFloat  = float2Double

-- It is necessary to propagate the size along the layers,
-- because fullconnect and convolution need to know
-- the previous size.
data LayerSize = D1 Int | D2 Int Int Int | SV LayerSize | SF Int LayerSize

-- 'InputLayer' is class for the input layer
class InputLayer i where
  isize :: i -> LayerSize

-- 'BodySize' is class for the actual computational layers
class BodySize l where
  bsize :: LayerSize -> l -> LayerSize

-- | Abstraction of a neural network component
class (Typeable a, Applicative (Run a)) => Component (a :: * -> *) where
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
  forwardT :: (Optimizer o, Data (a o)) => a o -> Inp a -> Run a (Trace a)
  -- | Forward propagation
  forward  :: (Optimizer o, Data (a o)) => a o -> Inp a -> Run a (Out a)
  forward a = (output <$>) . forwardT a
  -- | extract the output value from the trace
  output   :: Trace a -> Out a
  -- | Backward propagation
  backward :: (Optimizer o, Data (a o)) => a o -> Trace a -> Out a -> Run a (a o, Inp a)

class Evaluator a where
  type Val a
  eval :: a -> Val a -> IO (Val a)
  cost :: a -> Val a -> Val a -> IO (Val a)

-- translate the body of specification
class MonadError ErrCode (Env b) => BodyTrans b s where
  type SpecToCom b s :: * -> *
  -- The Run actually don't require opt, so
  -- + SpecToRun b s and a witness that SpecToRun b s ~ Run (SpecToCom b s o)
  -- + or, redefine Component (a :: * -> *), where opt is abstracted away
  -- type SpecToRun b s
  bwitness :: Optimizer o => b -> o -> s -> Dict (Component (SpecToCom b s), Data (SpecToCom b s o))
  btrans   :: (Optimizer o, Optimizable b o) => b -> o -> LayerSize -> s -> Env b (SpecToCom b s o)

data ErrCode = ErrMismatch

type ModelCst a b = (Component a, MonadIO (Run a), Evaluator b, Out a ~ Val b)

-- | Abstraction of backend to carry out the specification
class Backend b where
  -- | environment to 'compile' the specification
  type Env b :: * -> *
  -- | constraints of optimizable type, restricting the possible optimizer passed to 'compile'
  type Optimizable b o :: Constraint
  -- | result type of 'compile'
  type Compilable  b s e     :: Constraint
  type CompileComponent b s :: * -> *
  type CompileEvaluator b s :: *
  type CompileOptimizer b o :: *
  -- | necessary constraints of the resulting type
  witness :: (Optimizer o, Compilable b s e) =>
             b -> o -> e -> s -> Dict (Monad (Env b),
                                       ModelCst (CompileComponent b s) (CompileEvaluator b e))
  -- | compile the specification to runnable component.
  compile :: (InputLayer i, Compilable b s e, Optimizer o, Optimizable b o) =>
             b -> o -> e -> i -> s -> Env b ((CompileComponent b s o), (CompileEvaluator b e))
  -- | make a optimizer
  mkopt   :: b -> o -> Env b (CompileOptimizer b o)
