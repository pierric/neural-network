------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS.Layers
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
{-# LANGUAGE StandaloneDeriving #-}
module Data.NeuralNetwork.Backend.BLASHS.Layers(
  SinglVec, MultiMat, F, C, P, T, RunLayer(..),
  Reshape2DAs1D, as1D,
  newFLayer, newCLayer
) where

import qualified Data.Vector as V
import Blas.Generic.Unsafe (Numeric)
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2, forM_, when)
import GHC.Float
import Data.Data
import Data.STRef
import Data.NeuralNetwork
import Data.NeuralNetwork.Adapter
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD

type M = IO

-- | We parameterise the activation layer T, where the parameter indicates how
-- elements are contained:
data SinglVec
data MultiMat

-- | tag for the full-connect component
data F
-- | tag for the convolution component
data C
-- | tag for the max-pooling component
data P
-- | tag for the activation component
data T c

-- | basic components of neural network
data RunLayer :: * -> * -> * where
  -- | Densely connected layer
  -- input:   vector of size m
  -- output:  vector of size n
  -- weights: matrix of size m x n
  -- biases:  vector of size n
  Full :: !(DenseMatrix p) -> !(DenseVector p) -> RunLayer p F
  -- | Convolutional layer
  -- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
  -- output: channels of 2D floats, of the same size (c x d), # of output channels: n
  --         where c = a + 2*padding + 1 - s
  --               d = b + 2*padding + 1 - t
  -- feature:  matrix of (s x t), # of features: m x n
  -- padding:  number of 0s padded at each side of channel
  -- biases:   bias for each output, # of biases: n
  Conv  :: !(V.Vector (DenseMatrixArray p)) -> !(V.Vector p) -> Int -> RunLayer p C
  -- | max pooling layer
  -- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
  --         assuming that a and b are both multiple of stride
  -- output: channels of 2D floats, of the same size (c x d), # of output channels: m
  --         where c = a / stride
  --               d = b / stride
  MaxP :: Int -> RunLayer p P
  -- | Activator
  -- the input can be either a 1D vector, 2D matrix, or channels of either.
  Activation :: (SIMDPACK p -> SIMDPACK p, SIMDPACK p -> SIMDPACK p) -> RunLayer p (T c)

deriving instance Typeable (RunLayer p F)
deriving instance Typeable (RunLayer p C)
deriving instance Typeable (RunLayer p P)
deriving instance Typeable (RunLayer p (T SinglVec))
deriving instance Typeable (RunLayer p (T MultiMat))

instance Data p => Data (RunLayer p F) where
  toConstr (Full _ _) = fullConstr
  gfoldl f z (Full u v) = z (Full u v)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(RunLayer F)"
  dataTypeOf _  = runlayerDataType
instance Data p => Data (RunLayer p C) where
  toConstr (Conv _ _ _) = convConstr
  gfoldl f z (Conv u v w) = z (Conv u v w)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(RunLayer C)"
  dataTypeOf _  = runlayerDataType
instance Data p => Data (RunLayer p P) where
  toConstr (MaxP _) = maxpConstr
  gfoldl f z (MaxP u) = z (MaxP u)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(RunLayer P)"
  dataTypeOf _  = runlayerDataType
instance (Data p, Typeable a) => Data (RunLayer p (T a)) where
  toConstr (Activation _) = actiConstr
  gfoldl f z (Activation u) = z (Activation u)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(RunLayer T)"
  dataTypeOf _  = runlayerDataType

fullConstr = mkConstr runlayerDataType "Full" ["weights", "biases"] Prefix
convConstr = mkConstr runlayerDataType "Conv" ["kernels", "biases", "padding"] Prefix
maxpConstr = mkConstr runlayerDataType "MaxP" ["stride"] Prefix
actiConstr = mkConstr runlayerDataType "Activation" ["activation function"] Prefix
runlayerDataType = mkDataType "Data.NeuralNetwork.Backend.BLASHS.Utils.RunLayer"
                              [fullConstr, convConstr, maxpConstr, actiConstr]

instance (Numeric p, RealType p, SIMDable p) => Component (RunLayer p F) where
    type Dty (RunLayer p F) = p
    type Run (RunLayer p F) = IO
    type Inp (RunLayer p F) = DenseVector p
    type Out (RunLayer p F) = DenseVector p
    -- trace is (input, weighted-sum)
    newtype Trace (RunLayer p F) = DTrace (DenseVector p, DenseVector p)
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
        odelta <<= Scale (fromFloat $ negate rate)
        w <<+ iv :## odelta
        b <<= b  :.+ odelta
        return (Full w b, idelta)

instance (Numeric p, RealType p, SIMDable p) => Component (RunLayer p C) where
  type Dty (RunLayer p C) = p
  type Run (RunLayer p C) = IO
  type Inp (RunLayer p C) = V.Vector (DenseMatrix p)
  type Out (RunLayer p C) = V.Vector (DenseMatrix p)
  -- trace is (input, convoluted output)
  newtype Trace (RunLayer p C) = CTrace (Inp (RunLayer p C), Out (RunLayer p C))
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
        rate'   = fromFloat rate
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
                                  return $ b + negate rate' * s
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
                  corr2 qd odelta i ((fs <<+) . Scale' (negate rate'))
                ) fss iv
    let !ideltaV = denseMatrixArrayToVector idelta
    return $ (Conv fss nb pd, ideltaV)

instance (Numeric p, RealType p, SIMDable p) => Component (RunLayer p (T SinglVec)) where
    type Dty (RunLayer p (T SinglVec)) = p
    type Run (RunLayer p (T SinglVec)) = IO
    type Inp (RunLayer p (T SinglVec)) = DenseVector p
    type Out (RunLayer p (T SinglVec)) = DenseVector p
    newtype Trace (RunLayer p (T SinglVec)) = TTraceS (DenseVector p, DenseVector p)
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

instance (Numeric p, RealType p, SIMDable p) => Component (RunLayer p (T MultiMat)) where
    type Dty (RunLayer p (T MultiMat)) = p
    type Run (RunLayer p (T MultiMat)) = IO
    type Inp (RunLayer p (T MultiMat)) = V.Vector (DenseMatrix p)
    type Out (RunLayer p (T MultiMat)) = V.Vector (DenseMatrix p)
    newtype Trace (RunLayer p (T MultiMat)) = TTraceM (V.Vector (DenseMatrix p), V.Vector (DenseMatrix p))
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

instance (Numeric p, RealType p, SIMDable p) => Component (RunLayer p P) where
  type Dty (RunLayer p P) = p
  type Run (RunLayer p P) = IO
  type Inp (RunLayer p P) = V.Vector (DenseMatrix p)
  type Out (RunLayer p P) = V.Vector (DenseMatrix p)
  -- trace is (dimension of pools, index of max in each pool, pooled matrix)
  -- for each channel.
  newtype Trace (RunLayer p P) = PTrace (V.Vector ((Int,Int), DenseVector Int, DenseMatrix p))
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

-- | Reshape from channels of matrix to a single vector
-- input:  m channels of 2D matrices
--         assuming that all matrices are of the same size a x b
-- output: 1D vector of the concatenation of all input channels
--         its size: m x a x b
type Reshape2DAs1D p = Adapter IO (V.Vector (DenseMatrix p)) (DenseVector p) (Int, Int, Int)
as1D :: (Numeric p, RealType p, SIMDable p) => Reshape2DAs1D p
as1D = Adapter to back
  where
    to inp = do
      let !b = V.length inp
          (!r,!c) = size (V.head inp)
      o <- denseVectorConcat $ V.map m2v inp
      return ((b,r,c),o)
    back (b,r,c) odelta =
      return $! V.map (v2m r c) $ denseVectorSplit b (r*c) odelta



-- | create a new full connect component
newFLayer :: (Numeric p, RealType p, SIMDable p)
          => Int                -- ^ number of input values
          -> Int                -- ^ number of neurons (output values)
          -> IO (RunLayer p F)  -- ^ the new layer
newFLayer m n =
    withSystemRandom . asGenIO $ \gen -> do
        raw <- newDenseVectorByGen (fromDouble <$> normal 0 0.01 gen) (m*n)
        let w = v2m m n raw
        b <- newDenseVectorConst n 1
        return $ Full w b

-- | create a new convolutional component
newCLayer :: (Numeric p, RealType p, SIMDable p)
          => Int                -- ^ number of input channels
          -> Int                -- ^ number of output channels
          -> Int                -- ^ size of each feature
          -> Int                -- ^ size of padding
          -> IO (RunLayer p C)  -- ^ the new layer
newCLayer inpsize outsize sfilter npadding =
  withSystemRandom . asGenIO $ \gen -> do
      fss <- V.replicateM inpsize $ do
              raw <- newDenseVectorByGen (fromDouble <$> truncNormal 0 0.1 gen) (outsize*sfilter*sfilter)
              return $ v2ma outsize sfilter sfilter raw
      bs <- return $ V.replicate outsize 0.1
      return $ Conv fss bs npadding
  where
    truncNormal m s g = do
      x <- standard g
      if x >= 2.0 || x <= -2.0
        then truncNormal m s g
        else return $! m + s * x
