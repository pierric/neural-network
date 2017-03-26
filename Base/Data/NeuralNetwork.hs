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
module Data.NeuralNetwork (
  learn,
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
  SpecOptimizer(..),
  module Data.NeuralNetwork.Common
) where

import Control.Monad.Trans (liftIO)
import Data.Data
import Data.NeuralNetwork.Common

-- | By giving a way to measure the error, 'learn' can update the
-- neural network component.
learn :: (ModelCst n e, Optimizer o, Data (n o))
      => (n o,e)                            -- ^ neuron network
      -> (Inp n, Val e)                     -- ^ input and expect output
      -> Run n (n o,e)                      -- ^ updated network
learn (n,e) (i,o) = do
    tr <- forwardT n i
    o' <- liftIO $ eval e (output tr)
    er <- liftIO $ cost e o' o
    n' <- fst <$> backward n tr er
    return (n', e)

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
  deriving (Typeable, Data)

data SpecOptimizer = ScaleGrad Float | ADAGrad | ADAM
  deriving (Typeable, Data)

instance InputLayer SpecInStream where
  isize (InStream n) = SV (D1 n)
instance InputLayer SpecIn1D where
  isize (In1D n) = D1 n
instance InputLayer SpecIn2D where
  isize (In2D m n) = D2 1 m n

instance BodySize SpecReshape2DAs1D where
  bsize (D2 k m n) _ = D1 (k*m*n)
instance BodySize SpecFullConnect where
  bsize _ (FullConnect n) = D1 n
instance BodySize SpecConvolution where
  bsize (D2 _ m n) (Convolution k f p) = D2 k (m+2*p-f+1) (n+2*p-f+1)
instance BodySize SpecMaxPooling where
  bsize (D2 k m n) (MaxPooling s) = D2 k (m `div` s) (n `div` s)
instance BodySize SpecLSTM where
  bsize (D1 _) (LSTM n)= D1 n
instance BodySize a => BodySize (SpecFlow a) where
  bsize (SV sz) (Flow a) = SV (bsize sz a)
  bsize (SF n sz) (Flow a) = SF n (bsize sz a)
