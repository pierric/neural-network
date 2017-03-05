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
{-# LANGUAGE TypeOperators #-}
module Data.NeuralNetwork (
  Component(..),
  Evaluator(..),
  learn,
  Backend(..),
  RunInEnv(..),
  SpecIn1D(..),
  SpecIn2D(..),
  SpecInStream(..),
  SpecReshape2DAs1D(..),
  SpecFullConnect(..),
  SpecConvolution(..),
  SpecMaxPooling(..),
  SpecLSTM(..),
  SpecFlow(..),
  SpecMeanPooling(..),
  SpecEvaluator(..),
  Model,
) where

import Data.Data
import Data.Constraint

-- | Abstraction of a neural network component
class Component a where
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

class Monad m => Evaluator m a where
  type Val a
  eval :: a -> Val a -> m (Val a)
  cost :: a -> Val a -> Val a -> m (Val a)

-- | By giving a way to measure the error, 'learn' can update the
-- neural network component.
learn :: (Component n, Monad (Run n), Evaluator (Run n) e, Out n ~ Val e)
    => (n, e)                             -- ^ neuron network
    -> (Inp n, Val e)                     -- ^ input and expect output
    -> Float                              -- ^ learning rate
    -> Run n n                            -- ^ updated network
learn (n,e) (i,o) rate = do
    tr <- forwardT n i
    o' <- eval e (output tr)
    er <- cost e o' o
    fst <$> backward n tr er rate

{--
We can improve the predefined specifications by a feature like
"open Kinds", when it is ready:
https://ghc.haskell.org/trac/ghc/wiki/GhcKinds/KindsWithoutData

data kind open SpecKind
data kind member SpecIn1D :: SpecKind
data kind member SpecFlow :: SpecKind -> SpecKind
...

data family Specification :: SpecKind -> *
data instance Specification SpecIn1D = In1D Int
data instance Specification (SpecFlow a) = Flow (Specification a)
...

class Backend b (Specification s) where
  type Env b :: * -> *
  type ConvertFromSpec b s :: *
  compile :: b -> Specification s -> Env b (ConvertFromSpec b s)

The Major benefit would be that compiler could tell more if there
is an error when inferring the compiled neural network type.
--}

-- | Specification: 1D input
data SpecIn1D          = In1D Int     -- ^ dimension of input
  deriving (Typeable, Data)

-- | Specification: 2D input
data SpecIn2D          = In2D Int Int -- ^ dimension of input
  deriving (Typeable, Data)

data SpecInStream      = InStream Int
  deriving (Typeable, Data)

-- | Specification: full connection layer
data SpecFullConnect   = FullConnect Int  -- ^ number of neurals
  deriving (Typeable, Data)

-- | Specification: convolution layer
data SpecConvolution   = Convolution Int Int Int -- ^ number of output channels, size of kernel, size of padding
  deriving (Typeable, Data)

-- | Specification: max pooling layer
data SpecMaxPooling    = MaxPooling  Int
  deriving (Typeable, Data)

-- | Specification: max pooling layer
data SpecMeanPooling   = MeanPooling  Int
  deriving (Typeable, Data)

-- | Specification: reshaping layer
data SpecReshape2DAs1D = Reshape2DAs1D
  deriving (Typeable, Data)

data SpecLSTM = LSTM Int
  deriving (Typeable, Data)

data SpecFlow a = Flow a
  deriving (Typeable, Data)

data SpecEvaluator = MeanSquaredError | SoftmaxCrossEntropy

type family Model a where
  Model (a,b) = (Component a, Monad (Run a), Evaluator (Run a) b, Out a ~ Val b)
type family ModelRunInEnv m e where
  ModelRunInEnv (a,b) e = RunInEnv (Run a) e

-- | Abstraction of backend to carry out the specification
class Backend b s where
  -- | environment to 'compile' the specification
  type Env b :: * -> *
  -- | result type of 'compile'
  type ConvertFromSpec b s :: *
  -- | necessary constraints of the resulting type
  witness :: b -> s -> Dict ( Monad (Env b)
                            , Model (ConvertFromSpec b s)
                            , ModelRunInEnv (ConvertFromSpec b s) (Env b))
  -- | compile the specification to runnable component.
  compile :: b -> s -> Env b (ConvertFromSpec b s)

-- | Lifting from one monad to another.
-- It is not necessary that the 'Env' and 'Run' maps to the
-- same execution environment, but the 'Run' one should be
-- able to be lifted to 'Env' one.
class (Monad r, Monad e) => RunInEnv r e where
  run :: r a -> e a
