------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS.Utils
-- Description :  A backend for neuralnetwork with blas-hs.
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module supplies a high level abstraction of the rather
-- low-level blas-hs interfaces.
------------------------------------------------------------
{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts, GADTs #-}
module Data.NeuralNetwork.Backend.BLASHS.Layers(
  SinglVec, MultiMat, F, C, A, P, T, S, RunLayer(..),
  newFLayer, newCLayer
) where

import qualified Data.Vector as V
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2, forM_, when)
import GHC.Float
import Data.STRef
import Data.NeuralNetwork
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD

type R = Float
type M = IO

-- | We parameterise the activation layer T, where the parameter indicates how
-- elements are contained:
data SinglVec
data MultiMat

-- | tag for the full-connect component
data F
-- | tag for the convolution component
data C
-- | tag for the component that converts 2D as 1D
data A
-- | tag for the max-pooling component
data P
-- | tag for the activation component
data T c
-- | tag for the stacking component
data S a b

-- | basic components of neural network
data RunLayer :: * -> * where
  -- | Densely connected layer
  -- input:   vector of size m
  -- output:  vector of size n
  -- weights: matrix of size m x n
  -- biases:  vector of size n
  Full :: !(DenseMatrix R) -> !(DenseVector R) -> RunLayer F
  -- | Convolutional layer
  -- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
  -- output: channels of 2D floats, of the same size (c x d), # of output channels: n
  --         where c = a + 2*padding + 1 - s
  --               d = b + 2*padding + 1 - t
  -- feature:  matrix of (s x t), # of features: m x n
  -- padding:  number of 0s padded at each side of channel
  -- biases:   bias for each output, # of biases: n
  Conv  :: !(V.Vector (DenseMatrixArray R)) -> !(V.Vector R) -> Int -> RunLayer C
  -- | Reshape from channels of matrix to a single vector
  -- input:  m channels of 2D matrices
  --         assuming that all matrices are of the same size a x b
  -- output: 1D vector of the concatenation of all input channels
  --         its size: m x a x b
  As1D  :: RunLayer A
  -- | max pooling layer
  -- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
  --         assuming that a and b are both multiple of stride
  -- output: channels of 2D floats, of the same size (c x d), # of output channels: m
  --         where c = a / stride
  --               d = b / stride
  MaxP :: Int -> RunLayer P
  -- | Activator
  -- the input can be either a 1D vector, 2D matrix, or channels of either.
  Activation :: (SIMDPACK R -> SIMDPACK R, SIMDPACK R -> SIMDPACK R) -> RunLayer (T c)
  -- | stacking two components a and b
  -- the output of a should matches the input of b
  Stack :: !(RunLayer a) -> !(RunLayer b) -> RunLayer (S a b)

instance Component (RunLayer F) where
    type Run (RunLayer F) = IO
    type Inp (RunLayer F) = DenseVector R
    type Out (RunLayer F) = DenseVector R
    -- trace is (input, weighted-sum)
    newtype Trace (RunLayer F) = DTrace (DenseVector R, DenseVector R)
    forwardT (Full !w !b) !inp = do
        bv <- newDenseVectorCopy b
        bv <<+ inp :<# w
        return $ DTrace (inp,bv)
    output (DTrace (_,!a)) = a
    backward (Full !w !b) (DTrace (!iv,!bv)) !odelta rate = do
        -- back-propagated error at input
        idelta <- newDenseVector (fst $ size w)
        idelta <<= w :#> odelta
        -- odelta is not used any more, so we reuse it for an intermediate value.
        odelta <<= Scale (negate rate)
        w <<+ iv :## odelta
        b <<= b  :.+ odelta
        return (Full w b, idelta)

instance Component (RunLayer A) where
  type Run (RunLayer A) = IO
  type Inp (RunLayer A) = V.Vector (DenseMatrix R)
  type Out (RunLayer A) = DenseVector R
  -- trace keeps information of (m, a, b, output)
  newtype Trace (RunLayer A) = ReshapeTrace (Int, Int, Int, DenseVector R)
  forwardT _ !inp = do
    let !b = V.length inp
        (!r,!c) = size (V.head inp)
    o <- denseVectorConcat $ V.map m2v inp
    return $ ReshapeTrace (b, r, c, o)
  output (ReshapeTrace (_,_,_,a)) = a
  backward a (ReshapeTrace (b,r,c,_)) !odelta _ = do
    let !idelta = V.map (v2m r c) $ denseVectorSplit b (r*c) odelta
    return $ (a, idelta)

instance Component (RunLayer C) where
  type Run (RunLayer C) = IO
  type Inp (RunLayer C) = V.Vector (DenseMatrix R)
  type Out (RunLayer C) = V.Vector (DenseMatrix R)
  -- trace is (input, convoluted output)
  newtype Trace (RunLayer C) = CTrace (Inp (RunLayer C), Out (RunLayer C))
  forwardT (Conv fss bs pd) !inp = do
    ma <- newDenseMatrixArray outn outr outc
    V.zipWithM_ (\fs i -> corr2 pd (denseMatrixArrayToVector fs) i (ma <<+)) fss inp
    let ov = denseMatrixArrayToVector ma
    V.zipWithM_ (\m b -> m <<= Apply (plus (konst b))) ov bs
    return $ CTrace (inp, ov)
    where
      outn = V.length bs
      (outr,outc) = let (x,y)   = size (V.head inp)
                        (_,u,v) = size (V.head fss)
                    in (x+2*pd-u+1, y+2*pd-v+1)
  output (CTrace (_,!a)) = a
  backward (Conv fss bs pd) (CTrace (iv, av)) !odelta rate = do
    let (ir,ic) = size (V.head iv)
    idelta <- newDenseMatrixArray (V.length iv) ir ic
    fss'   <- transpose fss
    -- a + 2p - k + 1 = b
    -- b + 2q - a + 1 = a
    -- -------------------
    --    q = k - p
    -- where
    --   a = |i|, k = |f|, b = |o|
    let qd = let (kr,_) = size (V.head $ V.head fss') in kr-1-pd
    V.zipWithM_ (\fs d -> conv2 qd fs d (idelta <<+)) fss' odelta
    !nb <- V.zipWithM (\b d -> do s <- sumElements d
                                  return $ b + negate rate * s
                      ) bs odelta
    -- when updating kernels, it originally should be
    -- conv2 pd iv od. But we use the equalivalent form
    -- conv2 (|od|-|iv|+pd) od iv. Because there are typically
    -- more output channels than input.
    -- a + 2p - b + 1 = c
    -- b + 2q - a + 1 = c
    -- ------------------
    --    q = a - b + p
    -- where
    --  a = |o|, b = |i|, c = |f|
    let qd = let (or,_) = size (V.head odelta) in or - ir + pd
    V.zipWithM_ (\fs i -> do
                  -- i:  one input channel
                  -- fs: all features used for chn
                  corr2 qd odelta i ((fs <<+) . Scale' (negate rate))
                ) fss iv
    let !ideltaV = denseMatrixArrayToVector idelta
    return $ (Conv fss nb pd, ideltaV)
instance (Component (RunLayer a),
          Component (RunLayer b),
          Run (RunLayer a) ~ IO,
          Run (RunLayer b) ~ IO,
          Out (RunLayer a) ~ Inp (RunLayer b)
         ) => Component (RunLayer (S a b)) where
    type Run (RunLayer (S a b)) = IO
    type Inp (RunLayer (S a b)) = Inp (RunLayer a)
    type Out (RunLayer (S a b)) = Out (RunLayer b)
    newtype Trace (RunLayer (S a b)) = TTrace (Trace (RunLayer b), Trace (RunLayer a))
    forwardT (Stack a b) !i = do
        !tra <- forwardT a i
        !trb <- forwardT b (output tra)
        return $ TTrace (trb, tra)
    output (TTrace !a) = output (fst a)
    backward (Stack a b) (TTrace (!trb,!tra)) !odeltb rate = do
        (b', !odelta) <- backward b trb odeltb rate
        (a', !idelta) <- backward a tra odelta rate
        return (Stack a' b', idelta)

instance Component (RunLayer (T SinglVec)) where
    type Run (RunLayer (T SinglVec)) = IO
    type Inp (RunLayer (T SinglVec)) = DenseVector R
    type Out (RunLayer (T SinglVec)) = DenseVector R
    newtype Trace (RunLayer (T SinglVec)) = TTraceS (DenseVector R, DenseVector R)
    forwardT (Activation (af,_)) !inp = do
      out <- newDenseVectorCopy inp
      out <<= Apply af
      return $ TTraceS (inp, out)
    output (TTraceS (_,!a)) = a
    backward a@(Activation (_,ag)) (TTraceS (!iv,_)) !odelta _ = do
      idelta <- newDenseVectorCopy iv
      idelta <<= Apply ag
      idelta <<= odelta :.* idelta
      return $ (a, idelta)

instance Component (RunLayer (T MultiMat)) where
    type Run (RunLayer (T MultiMat)) = IO
    type Inp (RunLayer (T MultiMat)) = V.Vector (DenseMatrix R)
    type Out (RunLayer (T MultiMat)) = V.Vector (DenseMatrix R)
    newtype Trace (RunLayer (T MultiMat)) = TTraceM (V.Vector (DenseMatrix R), V.Vector (DenseMatrix R))
    forwardT (Activation (af,_)) !inp = do
      out <- V.mapM (\i -> do o <- newDenseMatrixCopy i
                              o <<= Apply af
                              return o
                    ) inp
      return $ TTraceM (inp, out)
    output (TTraceM (_,!a)) = a
    backward a@(Activation (_,ag)) (TTraceM (!iv,_)) !odelta _ = do
      idelta <- V.zipWithM (\i d -> do o <- newDenseMatrixCopy i
                                       o <<= Apply ag
                                       o <<= d :.* o
                                       return o
                           ) iv odelta
      return $ (a, idelta)

instance Component (RunLayer P) where
  type Run (RunLayer P) = IO
  type Inp (RunLayer P) = V.Vector (DenseMatrix R)
  type Out (RunLayer P) = V.Vector (DenseMatrix R)
  -- trace is (dimension of pools, index of max in each pool, pooled matrix)
  -- for each channel.
  newtype Trace (RunLayer P) = PTrace (V.Vector ((Int,Int), DenseVector Int, DenseMatrix R))
  -- forward is to divide the input matrix in stride x stride sub matrices,
  -- and then find the max element in each sub matrices.
  forwardT (MaxP stride) !inp = V.mapM mk inp >>= return . PTrace
    where
      mk inp = do
        (!i,!v) <- pool stride inp
        return (size v, i, v)
  output (PTrace a) = V.map (\(_,_,!o) ->o) a
  -- use the saved index-of-max in each pool to propagate the error.
  backward l@(MaxP stride) (PTrace t) odelta _ = do
      !idelta <- V.zipWithM gen t odelta
      return $ (l, idelta)
    where
      gen (!si,!iv,_) od = unpool stride iv od

-- | create a new full connect component
newFLayer :: Int                -- ^ number of input values
          -> Int                -- ^ number of neurons (output values)
          -> IO (RunLayer F)    -- ^ the new layer
newFLayer m n =
    withSystemRandom . asGenIO $ \gen -> do
        raw <- newDenseVectorByGen (double2Float <$> normal 0 0.01 gen) (m*n)
        let w = v2m m n raw
        b <- newDenseVectorConst n 1
        return $ Full w b

-- | create a new convolutional component
newCLayer :: Int                -- ^ number of input channels
          -> Int                -- ^ number of output channels
          -> Int                -- ^ size of each feature
          -> Int                -- ^ size of padding
          -> IO (RunLayer C)    -- ^ the new layer
newCLayer inpsize outsize sfilter npadding =
  withSystemRandom . asGenIO $ \gen -> do
      fss <- V.replicateM inpsize $ do
              raw <- newDenseVectorByGen (double2Float <$> truncNormal 0 0.1 gen) (outsize*sfilter*sfilter)
              return $ v2ma outsize sfilter sfilter raw
      bs <- return $ V.replicate outsize 0.1
      return $ Conv fss bs npadding
  where
    truncNormal m s g = do
      x <- standard g
      if x >= 2.0 || x <= -2.0
        then truncNormal m s g
        else return $! m + s * x