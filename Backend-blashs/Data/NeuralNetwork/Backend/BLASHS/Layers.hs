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
module Data.NeuralNetwork.Backend.BLASHS.Layers(
  FullConn(..), Convolute(..), MaxPool(..), ActivateS(..), ActivateM(..),
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
-- | Densely connected layer
-- input:   vector of size m
-- output:  vector of size n
-- weights: matrix of size m x n
-- biases:  vector of size n
data FullConn  o p = FullConn {
  _fc_weights :: !(WithVar DenseMatrix o p),
  _fc_bias    :: !(WithVar DenseVector o p)
} deriving Typeable
-- | Convolutional layer
-- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
-- output: channels of 2D floats, of the same size (c x d), # of output channels: n
--         where c = a + 2*padding + 1 - s
--               d = b + 2*padding + 1 - t
-- feature:  matrix of (s x t), # of features: m x n
-- padding:  number of 0s padded at each side of channel
-- biases:   bias for each output, # of biases: n
data Convolute o p = Convolute {
  _cn_kernels :: !(V.Vector (WithVar DenseMatrixArray o p)),
  _cn_bias    :: !(V.Vector (WithVar Scalar o p)),
  _cn_padding :: Int
} deriving Typeable
-- | max pooling layer
-- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
--         assuming that a and b are both multiple of stride
-- output: channels of 2D floats, of the same size (c x d), # of output channels: m
--         where c = a / stride
--               d = b / stride
data MaxPool  p = MaxPool Int
  deriving (Typeable, Data)
-- | Activator
-- the input can be either a 1D vector, 2D matrix
data ActivateS p = ActivateS (SIMDPACK p -> SIMDPACK p) (SIMDPACK p -> SIMDPACK p)
  deriving Typeable
-- | Activator
-- the input can be channels of 1D vector or 2D matrix.
data ActivateM p = ActivateM (SIMDPACK p -> SIMDPACK p) (SIMDPACK p -> SIMDPACK p)
  deriving Typeable

instance (Data o, Data p) => Data (FullConn o p) where
  toConstr (FullConn _ _)   = fullConstr
  gfoldl f z (FullConn u v) = z (FullConn u v)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(FullConn)"
  dataTypeOf _  = fullType
instance (Data o, Data p) => Data (Convolute o p) where
  toConstr _ = convConstr
  gfoldl f z c = z c
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(Convolute)"
  dataTypeOf _  = convType
instance Data p => Data (ActivateS p) where
  toConstr (ActivateS _ _) = actsConstr
  gfoldl f z (ActivateS u v) = z (ActivateS u v)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(ActivateS)"
  dataTypeOf _  = actsType
instance Data p => Data (ActivateM p) where
  toConstr (ActivateM _ _) = actmConstr
  gfoldl f z (ActivateM u v) = z (ActivateM u v)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(ActivateM)"
  dataTypeOf _  = actmType

fullConstr = mkConstr fullType "FullConn"  ["weights", "biases"] Prefix
fullType   = mkDataType "Data.NeuralNetwork.Backend.BLASHS.Utils.FullConn" [fullConstr]
convConstr = mkConstr convType "Convolute" ["kernels", "biases", "padding"] Prefix
convType   = mkDataType "Data.NeuralNetwork.Backend.BLASHS.Utils.Convolute" [convConstr]
actsConstr = mkConstr actsType "ActivateS" ["activation forward", "activation backward"] Prefix
actsType   = mkDataType "Data.NeuralNetwork.Backend.BLASHS.Utils.ActivateS" [actsConstr]
actmConstr = mkConstr actmType "ActivateM" ["activation forward", "activation backward"] Prefix
actmType   = mkDataType "Data.NeuralNetwork.Backend.BLASHS.Utils.ActivateM" [actmConstr]

instance (Numeric p, RealType p, SIMDable p, Optimizer o) => Component (FullConn o p) where
    type Dty (FullConn o p) = p
    type Run (FullConn o p) = IO
    type Inp (FullConn o p) = DenseVector p
    type Out (FullConn o p) = DenseVector p
    -- trace is (input, weighted-sum)
    newtype Trace (FullConn o p) = DTrace (DenseVector p, DenseVector p)
    forwardT (FullConn !w !b) !inp = do
        bv <- newDenseVectorCopy (_parm b)
        bv <<+ inp :<# _parm w
        return $ DTrace (inp,bv)
    output (DTrace (_,!a)) = a
    backward (FullConn !w !b) (DTrace (!iv,!bv)) !odelta = do
        -- back-propagated error at input
        idelta <- newDenseVector (fst $ size $ _parm w)
        idelta <<= _parm w :#> odelta

        dw <- uncurry newDenseMatrix (size $ _parm w)
        dw <<= iv :## odelta
        dw <- optimize (_ovar w) dw
        _parm w <<= _parm w :.+ dw

        db <- optimize (_ovar b) odelta
        _parm b <<= _parm b  :.+ db
        return (FullConn w b, idelta)

instance (Numeric p, RealType p, SIMDable p, Optimizer o) => Component (Convolute o p) where
  type Dty (Convolute o p) = p
  type Run (Convolute o p) = IO
  type Inp (Convolute o p) = V.Vector (DenseMatrix p)
  type Out (Convolute o p) = V.Vector (DenseMatrix p)
  -- trace is (input, convoluted output)
  newtype Trace (Convolute o p) = CTrace (Inp (Convolute o p), Out (Convolute o p))
  forwardT (Convolute fss bs pd) !inp = do
    ma <- newDenseMatrixArray outn outr outc
    V.zipWithM_ (\fs i -> corr2 pd (denseMatrixArrayToVector $ _parm fs) i (ma <<+)) fss inp
    let ov = denseMatrixArrayToVector ma
    V.zipWithM_ (\m b -> m <<= Apply (plus (konst $ unScalar $ _parm b))) ov bs
    return $ CTrace (inp, ov)
    where
      outn = V.length bs
      (outr,outc) = let (x,y)   = size (V.head inp)
                        (_,u,v) = size (V.head fss)
                    in (x+2*pd-u+1, y+2*pd-v+1)
  output (CTrace (_,!a)) = a
  backward (Convolute fss bs pd) (CTrace (iv, av)) !odelta = do
    let (ir,ic) = size (V.head iv)
    idelta <- newDenseMatrixArray (V.length iv) ir ic
    fss'   <- transpose (V.map _parm fss)
    -- a + 2p - k + 1 = b
    -- b + 2q - a + 1 = a
    -- -------------------
    --    q = k - p
    -- where
    --   a = |i|, k = |f|, b = |o|
    let qd = let (kr,_) = size (V.head $ V.head fss') in kr-1-pd
    V.zipWithM_ (\fs d -> conv2 qd fs d (idelta <<+)) fss' odelta
    !nb <- V.forM (V.zip bs odelta) $ \(b, d) -> do
             s <- sumElements d
             s <- optimize (_ovar b) (Scalar s)
             return $ b{ _parm = _parm b + s}
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
    V.forM_ (V.zip fss iv) $ \(fs, i) -> do
      -- i:  one input channel
      -- fs: all features used for chn
      dfs <- let (c,m,n) = size (_parm fs) in newDenseMatrixArray c m n
      corr2 qd odelta i (dfs <<=)
      dfs <- optimize (_ovar fs) dfs
      _parm fs <<= _parm fs :.+ dfs

    let !ideltaV = denseMatrixArrayToVector idelta
    return $ (Convolute fss nb pd, ideltaV)

instance (Numeric p, RealType p, SIMDable p) => Component (ActivateS p) where
    type Dty (ActivateS p) = p
    type Run (ActivateS p) = IO
    type Inp (ActivateS p) = DenseVector p
    type Out (ActivateS p) = DenseVector p
    newtype Trace (ActivateS p) = TTraceS (DenseVector p, DenseVector p)
    forwardT (ActivateS af _) !inp = do
      out <- newDenseVectorCopy inp
      out <<= Apply af
      return $ TTraceS (inp, out)
    output (TTraceS (_,!a)) = a
    backward a@(ActivateS _ ag) (TTraceS (!iv,_)) !odelta = do
      idelta <- newDenseVectorCopy iv
      idelta <<= Apply ag
      idelta <<= odelta :.* idelta
      return $ (a, idelta)

instance (Numeric p, RealType p, SIMDable p) => Component (ActivateM p) where
    type Dty (ActivateM p) = p
    type Run (ActivateM p) = IO
    type Inp (ActivateM p) = V.Vector (DenseMatrix p)
    type Out (ActivateM p) = V.Vector (DenseMatrix p)
    newtype Trace (ActivateM p) = TTraceM (V.Vector (DenseMatrix p), V.Vector (DenseMatrix p))
    forwardT (ActivateM af _) !inp = do
      out <- V.mapM (\i -> do o <- newDenseMatrixCopy i
                              o <<= Apply af
                              return o
                    ) inp
      return $ TTraceM (inp, out)
    output (TTraceM (_,!a)) = a
    backward a@(ActivateM _ ag) (TTraceM (!iv,_)) !odelta = do
      idelta <- V.zipWithM (\i d -> do o <- newDenseMatrixCopy i
                                       o <<= Apply ag
                                       o <<= d :.* o
                                       return o
                           ) iv odelta
      return $ (a, idelta)

instance (Numeric p, RealType p, SIMDable p) => Component (MaxPool p) where
  type Dty (MaxPool p) = p
  type Run (MaxPool p) = IO
  type Inp (MaxPool p) = V.Vector (DenseMatrix p)
  type Out (MaxPool p) = V.Vector (DenseMatrix p)
  -- trace is (dimension of pools, index of max in each pool, pooled matrix)
  -- for each channel.
  newtype Trace (MaxPool p) = PTrace (V.Vector ((Int,Int), DenseVector Int, DenseMatrix p))
  -- forward is to divide the input matrix in stride x stride sub matrices,
  -- and then find the max element in each sub matrices.
  forwardT (MaxPool stride) !inp = V.mapM mk inp >>= return . PTrace
    where
      mk inp = do
        (!i,!v) <- pool stride inp
        return (size v, i, v)
  output (PTrace a) = V.map (\(_,_,!o) ->o) a
  -- use the saved index-of-max in each pool to propagate the error.
  backward l@(MaxPool stride) (PTrace t) odelta = do
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
newFLayer :: (Numeric p, RealType p, SIMDable p, Optimizer o)
          => Int                -- ^ number of input values
          -> Int                -- ^ number of neurons (output values)
          -> o
          -> IO (FullConn o p)    -- ^ the new layer
newFLayer m n opt =
    withSystemRandom . asGenIO $ \gen -> do
        raw <- newDenseVectorByGen (fromDouble <$> normal 0 0.01 gen) (m*n)
        let w = v2m m n raw
        ow  <- newOptVar opt w
        b   <- newDenseVectorConst n 1
        ob  <- newOptVar opt b
        return $ FullConn (WithVar w ow) (WithVar b ob)

-- | create a new convolutional component
newCLayer :: (Numeric p, RealType p, SIMDable p, Optimizer o)
          => Int                -- ^ number of input channels
          -> Int                -- ^ number of output channels
          -> Int                -- ^ size of each feature
          -> Int                -- ^ size of padding
          -> o
          -> IO (Convolute o p)   -- ^ the new layer
newCLayer inpsize outsize sfilter npadding opt =
  withSystemRandom . asGenIO $ \gen -> do
      fss <- V.replicateM inpsize $ do
              raw <- newDenseVectorByGen (fromDouble <$> truncNormal 0 0.1 gen) (outsize*sfilter*sfilter)
              let ma = v2ma outsize sfilter sfilter raw
              oma <- newOptVar opt ma
              return $ WithVar ma oma
      bs  <- V.replicateM outsize $ do
               let v = 0.1
               ov <- newOptVar opt v
               return $ WithVar v ov
      return $ Convolute fss bs npadding
  where
    truncNormal m s g = do
      x <- standard g
      if x >= 2.0 || x <= -2.0
        then truncNormal m s g
        else return $! m + s * x
