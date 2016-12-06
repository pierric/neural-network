{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts, GADTs #-}
module Data.NeuronNetwork.Backend.BLASHS.Layers where

import qualified Data.Vector as V
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2, forM_, when)
import GHC.Float
import Data.STRef
import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.BLASHS.Utils

type R = Float
type M = IO

-- We parameterise the activation layer T, where the parameter indicates how
-- elements are contained:
data SinglVec
data MultiMat

-- Tags for each form of layer
data F
data C
data A
data T c
data S a b

data RunLayer :: * -> * where
  -- Densely connected layer
  -- input:   vector of size m
  -- output:  vector of size n
  -- weights: matrix of size m x n
  -- biases:  vector of size n
  Full :: !(DenseMatrix R) -> !(DenseVector R) -> RunLayer F
  -- convolutional layer
  -- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
  -- output: channels of 2D floats, of the same size (c x d), # of output channels: n
  --         where c = a + 2*padding + 1 - s
  --               d = b + 2*padding + 1 - t
  -- feature:  matrix of (s x t), # of features: m x n
  -- padding:  number of 0s padded at each side of channel
  -- biases:   bias for each output, # of biases: n
  Conv  :: !(V.Vector (V.Vector (DenseMatrix R))) -> !(V.Vector R) -> Int -> RunLayer C
  -- Reshape from channels of matrix to a single vector
  -- input:  m channels of 2D matrices
  --         assuming that all matrices are of the same size a x b
  -- output: 1D vector of the concatenation of all input channels
  --         its size: m x a x b
  As1D  :: RunLayer A
  -- Activator
  -- the input can be either a 1D vector, 2D matrix, or channels of either.
  Activation :: (R->R, R->R) -> RunLayer (T c)
  -- stacking two components a and b
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
    o <- concatV $ map m2v $ V.toList inp
    return $ ReshapeTrace (b, r, c, o)
  output (ReshapeTrace (_,_,_,a)) = a
  backward a (ReshapeTrace (b,r,c,_)) !odelta _ =
    let !idelta = V.fromList $ map (v2m r c) $ splitV b (r*c) odelta
    in return $ (a, idelta)

instance Component (RunLayer C) where
  type Run (RunLayer C) = IO
  type Inp (RunLayer C) = V.Vector (DenseMatrix R)
  type Out (RunLayer C) = V.Vector (DenseMatrix R)
  -- trace is (input, convoluted output)
  newtype Trace (RunLayer C) = CTrace (Inp (RunLayer C), Out (RunLayer C))
  forwardT (Conv fs bs pd) !inp = do
    ov <- V.zipWithM feature (tr fs) bs
    return $ CTrace (inp, ov)
    where
      -- transpose the features matrix
      tr :: V.Vector (V.Vector a) -> V.Vector (V.Vector a)
      tr uv = let n   = V.length (V.head uv)
                  !vu = V.map (\i -> V.map (V.! i) uv) $ V.enumFromN 0 n
              in vu
      (outr,outc) = let (x,y) = size (V.head inp)
                        (u,v) = size (V.head $ V.head fs)
                    in (x+2*pd-u+1, y+2*pd-v+1)
      feature :: V.Vector (DenseMatrix R) -> R -> Run (RunLayer C) (DenseMatrix R)
      feature f b = do
        mat <- newDenseMatrix outr outc
        V.zipWithM_ (\a b -> corr2 pd a b (mat <<+)) f inp
        mat <<= Apply (+ b)
        return mat

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

newFLayer :: Int                -- number of input values
          -> Int                -- number of neurons (output values)
          -> IO (RunLayer F)    -- new layer
newFLayer m n =
    withSystemRandom . asGenIO $ \gen -> do
        -- we build the weights in column major because in the back-propagation
        -- algo, the computed update to weights is in column major. So it is
        -- good for performance to keep the matrix always in column major.
        w <- newDenseMatrixByGen (double2Float <$> normal 0 0.01 gen) m n
        b <- newDenseMatrixConst n 1
        return $ Full w b
