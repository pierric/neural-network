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
{-# LANGUAGE UndecidableInstances #-}
module Data.NeuralNetwork.Backend.BLASHS (
  -- module Data.NeuralNetwork.Backend.BLASHS.Layers,
  module Data.NeuralNetwork.Backend.BLASHS.Utils,
  module Data.NeuralNetwork.Backend.BLASHS.LSTM,
  ByBLASHS(..),
  ErrCode(..),
  cost',
) where

import Data.NeuralNetwork hiding (relu, relu', cost')
import Data.NeuralNetwork.Stack
import Data.NeuralNetwork.Common
import Data.NeuralNetwork.Backend.BLASHS.Layers
import Data.NeuralNetwork.Backend.BLASHS.LSTM
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
instance (HeadSize z, TranslateBody Err s,
          Component (SpecToCom s),
          RunInEnv (Run (SpecToCom s)) Err)
       => Backend ByBLASHS (z :++ s) where
  type Env ByBLASHS = Err
  type ConvertFromSpec ByBLASHS (z :++ s) = SpecToCom s
  compile _ (a :++ l)= trans (hsize a) l
  witness _ _ = Dict

instance RunInEnv IO Err where
  run = liftIO

instance TranslateBody Err SpecFullConnect where
  -- 'SpecFullConnect' is translated to a two-layer component
  -- a full-connect, followed by a relu activation (1D, single channel)
  type SpecToCom SpecFullConnect = Stack (RunLayer F) (RunLayer (T SinglVec)) CE
  trans (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                    return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody Err SpecConvolution where
  -- 'SpecConvolution' is translated to a two-layer component
  -- a convolution, following by a relu activation (2D, multiple channels)
  type SpecToCom SpecConvolution = Stack (RunLayer C) (RunLayer (T MultiMat)) CE
  trans (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p
                                            return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody Err SpecMaxPooling where
  -- 'MaxPooling' is translated to a max-pooling component.
  type SpecToCom SpecMaxPooling = RunLayer P
  trans (D2 _ _ _) (MaxPooling n) = return (MaxP n)
  trans _ _ = throwError ErrMismatch

instance TranslateBody Err SpecReshape2DAs1D where
  -- 'SpecReshape2DAs1D' is translated to a reshaping component.
  type SpecToCom SpecReshape2DAs1D = Reshape2DAs1D
  trans (D2 _ _ _) _ = return as1D
  trans _ _ = throwError ErrMismatch

instance TranslateBody Err SpecLSTM where
  -- 'SpecLSTM' is translated to a LSTM component.
  type SpecToCom SpecLSTM = Stack LSTM (RunLayer (T SinglVec)) (LiftRun (Run LSTM) (Run (RunLayer (T SinglVec))))
  trans (D1 s) (LSTM n) = do u <- lift $ newLSTM s n
                             return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance (TranslateBody Err a) => TranslateBody Err (SpecFlow a) where
  --
  type SpecToCom (SpecFlow a) = Stream (SpecToCom a)
  trans (SV s) (Flow a) = do u <- trans s a
                             return $ Stream u
  trans _ _ = throwError ErrMismatch
