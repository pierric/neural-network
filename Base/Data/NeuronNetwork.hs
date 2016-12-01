{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
module Data.NeuronNetwork (
  Component(..),
  learn,
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

class Component a where
  type Run a :: * -> *
  -- the input and in-error are typed by 'Inp a'
  type Inp a
  -- the output and out-error are typed by 'Out a'
  type Out a
  -- the trace start with the input, following
  -- by weighted-sum value and activated value
  -- of each inner level.
  data Trace a
  -- Forward propagation
  -- input: layer, input value
  -- output: a trace
  forwardT :: a -> Inp a -> Run a (Trace a)
  -- Forward propagation
  -- input: layer, input value
  -- output: the final activated value.
  forward  :: Applicative (Run a) => a -> Inp a -> Run a (Out a)
  forward a = (output <$>) . forwardT a
  -- extract the final activated value from the trace
  output   :: Trace a -> Out a
  -- input:  old layer, a trace, out-error, learning rate
  -- output: new layer, in-error
  backward :: a -> Trace a -> Out a -> Float -> Run a (a, Inp a)

learn :: (Component n, Monad (Run n))
    => (Out n -> Out n -> Out n)  -- derivative of the error function
    -> Float                      -- learning rate
    -> n                          -- neuron network
    -> (Inp n, Out n)             -- input and expect output
    -> Run n n                    -- updated network
learn cost rate n (i,o) = do
    tr <- forwardT n i
    fst <$> backward n tr (cost (output tr) o) rate

data SpecIn1D          = In1D Int
data SpecIn2D          = In2D Int Int
data SpecFullConnect   = FullConnect Int
data SpecConvolution   = Convolution Int Int Int
data SpecMaxPooling    = MaxPooling  Int
data SpecReshape2DAs1D = Reshape2DAs1D

infixr 0 :++
data a :++ b = a :++ b

class Backend b s where
  type Env b :: * -> *
  type ConvertFromSpec s :: *
  witness :: b -> s -> Dict (Monad (Env b), Monad (Run (ConvertFromSpec s)), Component (ConvertFromSpec s), RunInEnv (Run (ConvertFromSpec s))  (Env b))
  compile :: b -> s -> Env b (ConvertFromSpec s)

class RunInEnv r e where
  run :: r a -> e a
