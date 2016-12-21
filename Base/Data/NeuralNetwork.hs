{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
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
  SpecMaxPooling(..)
) where

import Data.Constraint

-- | Abstraction of a neural network component
class Component a where
  -- | execution environment
  type Run a :: * -> *
  -- | the input and in-error are typed by 'Inp a'
  type Inp a
  -- | the output and out-error are typed by 'Out a'
  type Out a
  -- | the trace of a forward propagation.
  data Trace a
  -- | Forward propagation
  -- input: layer, input value
  -- output: a trace
  forwardT :: a -> Inp a -> Run a (Trace a)
  -- | Forward propagation
  -- input: layer, input value
  -- output: the output value.
  forward  :: Applicative (Run a) => a -> Inp a -> Run a (Out a)
  forward a = (output <$>) . forwardT a
  -- | extract the output value from the trace
  output   :: Trace a -> Out a
  -- | Backward propagation
  -- input:  old layer, a trace, out-error, learning rate
  -- output: new layer, in-error
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

-- | Specification of neural network
data SpecIn1D          = In1D Int
data SpecIn2D          = In2D Int Int
data SpecFullConnect   = FullConnect Int
data SpecConvolution   = Convolution Int Int Int
data SpecMaxPooling    = MaxPooling  Int
data SpecReshape2DAs1D = Reshape2DAs1D
infixr 0 :++
data a :++ b = a :++ b

-- | Abstraction of backend to carry out the specification.
class Backend b s where
  -- | environment to 'compile' the specification
  type Env b :: * -> *
  -- | result type of 'compile'.
  type ConvertFromSpec s :: *
  -- | necessary constraints of the resulting type.
  witness :: b -> s -> Dict ( Monad (Env b)
                            , Monad (Run (ConvertFromSpec s))
                            , Component (ConvertFromSpec s)
                            , RunInEnv (Run (ConvertFromSpec s)) (Env b))
  -- | compile the specification to runnable component.
  compile :: b -> s -> Env b (ConvertFromSpec s)

-- | Lifting from one monad to another.
-- It is not necessary that the 'Env' and 'Run' maps to the
-- same execution environment, but the 'Run' one should be
-- able to be lifted to 'Env' one.
class (Monad r, Monad e) => RunInEnv r e where
  run :: r a -> e a
