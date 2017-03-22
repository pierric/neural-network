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
module Data.NeuralNetwork.Common(
  RealType(..),
  LayerSize(..),
  InputLayer(..),
  BodySize(..), BodyTrans(..), EvalTrans(..),
  ErrCode(..),
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
  HVect((:&:),HNil),
  OptVar,
  Optimizer(..),
) where

import Control.Monad.Except (MonadError)
import GHC.Float (double2Float, float2Double)
import Data.Data
import Data.HVect

data OptVar opt grad = OptVar opt [grad]

class Optimizer a where
  newOptVar :: a -> b -> IO (OptVar a b)
  optimize :: OptVar a b -> b -> IO b

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
instance InputLayer SpecInStream where
  isize (InStream n) = SV (D1 n)
instance InputLayer SpecIn1D where
  isize (In1D n) = D1 n
instance InputLayer SpecIn2D where
  isize (In2D m n) = D2 1 m n

-- 'BodySize' is class for the actual computational layers
class BodySize l where
  bsize :: LayerSize -> l -> LayerSize
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

-- translate the body of specification
class MonadError ErrCode m => BodyTrans m b s where
  type SpecToCom b o s
  btrans :: Optimizer o => b -> LayerSize -> s -> o -> m (SpecToCom b o s)

class (MonadError ErrCode m) => EvalTrans m b c s where
  type SpecToEvl b c s
  etrans :: b -> c -> s -> m (SpecToEvl b c s)

data ErrCode = ErrMismatch

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
