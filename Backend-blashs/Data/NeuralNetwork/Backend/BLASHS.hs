------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS
-- Description :  A backend for neural network on top of 'blas-hs'
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module supplies a backend for the neural-network-base
-- package. This backend is implemented on top of the blas-hs
-- package and optimised with SIMD.
------------------------------------------------------------
{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
module Data.NeuralNetwork.Backend.BLASHS (
  -- module Data.NeuralNetwork.Backend.BLASHS.Layers,
  module Data.NeuralNetwork.Backend.BLASHS.Utils,
  ByBLASHS(..),
  ErrCode(..),
  cost'
) where

import Data.NeuralNetwork hiding (relu, relu', cost')
import Data.NeuralNetwork.Stack
import Data.NeuralNetwork.Backend.BLASHS.Layers
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD
import Control.Monad.Except
import Control.Monad.State
import Data.Constraint (Dict(..))

-- | Compilation of the specification of a neural network is carried out in
-- the 'Err' monad, and the possible errors are characterized by 'ErrCode'.
type Err     = ExceptT ErrCode IO
data ErrCode = ErrMismatch

-- | The backend data type
data ByBLASHS = ByBLASHS

-- | Neural network specified to start with 1D / 2D input
instance (HeadSize z, TranslateBody s,
          Component (SpecToCom s),
          RunInEnv (Run (SpecToCom s)) Err)
       => Backend ByBLASHS (z :++ s) where
  type Env ByBLASHS = Err
  type ConvertFromSpec ByBLASHS (z :++ s) = SpecToCom s
  compile _ (a :++ l)= trans (hsize a) l
  witness _ _ = Dict

instance RunInEnv IO Err where
  run = liftIO

-- It is necessary to propagate the size along the layers,
-- because fullconnect and convolution need to know
-- the previous size.
data LayerSize = D1 Int | D2 Int Int Int

-- 'HeadSize' is class for the input layer
class HeadSize l where
  hsize :: l -> LayerSize
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

type family SpecToCom s where
  -- 'SpecFullConnect' is translated to a two-layer component
  -- a full-connect, followed by a relu activation (1D, single channel)
  SpecToCom SpecFullConnect = Stack (RunLayer F) (RunLayer (T SinglVec))
  -- 'SpecConvolution' is translated to a two-layer component
  -- a convolution, following by a relu activation (2D, multiple channels)
  SpecToCom SpecConvolution = Stack (RunLayer C) (RunLayer (T MultiMat))
  -- 'MaxPooling' is translated to a max-pooling component.
  SpecToCom SpecMaxPooling = RunLayer P
  -- 'SpecReshape2DAs1D' is translated to a reshaping component.
  SpecToCom SpecReshape2DAs1D = Reshape2DAs1D
  -- ':++' is translated to the stacking component.
  SpecToCom (a :++ b) = Stack (SpecToCom a) (SpecToCom b)

-- translate the body of specification
class TranslateBody s where
  trans :: LayerSize -> s -> Err (SpecToCom s)

instance TranslateBody SpecFullConnect where
  trans (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                    return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecConvolution where
  trans (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p
                                            return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecMaxPooling where
  trans (D2 _ _ _) (MaxPooling n) = return (MaxP n)
  trans (D1 _)     _              = throwError ErrMismatch

instance TranslateBody SpecReshape2DAs1D where
  trans (D2 _ _ _) _ = return as1D
  trans (D1 _)     _ = throwError ErrMismatch

instance (TranslateBody a, TranslateBody c, BodySize a) => TranslateBody (a :++ c) where
  trans s (a :++ c) = do u <- trans s a
                         v <- trans (bsize s a) c
                         return $ Stack u v
