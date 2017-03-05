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
  LayerSize(..),
  InputLayer(..),
  BodySize(..), BodyTrans(..), EvalTrans(..),
  ErrCode(..),
) where

import Control.Monad.Except (MonadError)
import Data.HVect
import Data.NeuralNetwork

-- It is necessary to propagate the size along the layers,
-- because fullconnect and convolution need to know
-- the previous size.
data LayerSize = D1 Int | D2 Int Int Int | SV LayerSize | SF Int LayerSize

-- 'HeadSize' is class for the input layer
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
class MonadError ErrCode m => BodyTrans m s where
  type SpecToCom s
  trans :: LayerSize -> s -> m (SpecToCom s)

class (MonadError ErrCode m, Component c) => EvalTrans m c e where
  type SpecToEvl c e
  etrans :: c -> e -> m (SpecToEvl c e)

data ErrCode = ErrMismatch
