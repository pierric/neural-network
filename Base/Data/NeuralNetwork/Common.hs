{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

module Data.NeuralNetwork.Common(
  LayerSize(..),
  HeadSize(..),
  BodySize(..),
  TranslateBody(..),
) where

import Data.NeuralNetwork

-- It is necessary to propagate the size along the layers,
-- because fullconnect and convolution need to know
-- the previous size.
data LayerSize = D1 Int | D2 Int Int Int | SV LayerSize | SF Int LayerSize

-- 'HeadSize' is class for the input layer
class HeadSize l where
  hsize :: l -> LayerSize
instance HeadSize SpecInStream where
  hsize InStream = SV (D1 1)
instance HeadSize SpecIn1D where
  hsize (In1D n) = D1 n
instance HeadSize SpecIn2D where
  hsize (In2D m n) = D2 1 m n

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
class TranslateBody m s where
  type SpecToCom s
  trans :: LayerSize -> s -> m (SpecToCom s)
