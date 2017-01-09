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
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveDataTypeable #-}
module Data.NeuralNetwork (
  Component(..),
  learn,
  relu, relu',
  cost',
  Backend(..),
  RunInEnv(..),
  (:++)(..),
  SpecIn1D(..),
  SpecIn2D(..),
  SpecReshape2DAs1D(..),
  SpecFullConnect(..),
  SpecConvolution(..),
  SpecMaxPooling(..),
  SpecLSTM(..)
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

-- | By giving a way to measure the error, 'learn' can update the
-- neural network component.
learn :: (Component n, Monad (Run n))
    => (Out n -> Out n -> Run n (Out n))  -- ^ derivative of the error function
    -> Float                              -- ^ learning rate
    -> n                                  -- ^ neuron network
    -> (Inp n, Out n)                     -- ^ input and expect output
    -> Run n n                            -- ^ updated network
learn cost rate n (i,o) = do
    tr <- forwardT n i
    er <- cost (output tr) o
    fst <$> backward n tr er rate

-- | default RELU and derivative of RELU
relu, relu' :: (Num a, Ord a) => a -> a
relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

-- | default derivative of error measurement
cost' :: (Num a, Ord a) => a -> a -> a
cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

-- | Specification: 1D input
data SpecIn1D          = In1D Int     -- ^ dimension of input
  deriving (Typeable, Data)

-- | Specification: 2D input
data SpecIn2D          = In2D Int Int -- ^ dimension of input
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

-- | Specification: reshaping layer
data SpecReshape2DAs1D = Reshape2DAs1D
  deriving (Typeable, Data)

data SpecLSTM = LSTM
  deriving (Typeable, Data)

-- | Specification: stacking layer
infixr 0 :++
data a :++ b = a :++ b
  deriving (Typeable, Data)

-- | Abstraction of backend to carry out the specification
class Backend b s where
  -- | environment to 'compile' the specification
  type Env b :: * -> *
  -- | result type of 'compile'
  type ConvertFromSpec b s :: *
  -- | necessary constraints of the resulting type
  witness :: b -> s -> Dict ( Monad (Env b)
                            , Monad (Run (ConvertFromSpec b s))
                            , Component (ConvertFromSpec b s)
                            , RunInEnv (Run (ConvertFromSpec b s)) (Env b))
  -- | compile the specification to runnable component.
  compile :: b -> s -> Env b (ConvertFromSpec b s)

-- | Lifting from one monad to another.
-- It is not necessary that the 'Env' and 'Run' maps to the
-- same execution environment, but the 'Run' one should be
-- able to be lifted to 'Env' one.
class (Monad r, Monad e) => RunInEnv r e where
  run :: r a -> e a
