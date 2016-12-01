{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts, GADTs #-}
module Data.NeuronNetwork.Backend.BLASHS.Layers where

import qualified Data.Vector as V
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2, forM_, when)
import GHC.Float
import Data.STRef
import Control.DeepSeq
import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.BLASHS.Utils

type R = Float
type M = IO

-- Tags for each form of layer
data F
data S a b

data RunLayer :: * -> * where
  -- Densely connected layer
  -- input:   vector of size m
  -- output:  vector of size n
  -- weights: matrix of size m x n
  -- biases:  vector of size n
  Full :: !(DenseMatrix R) -> !(DenseVector R) -> RunLayer F
  -- stacking two components a and b
  -- the output of a should matches the input of b
  Stack :: !(RunLayer a) -> !(RunLayer b) -> RunLayer (S a b)

instance Component (RunLayer F) where
    type Run (RunLayer F) = IO
    type Inp (RunLayer F) = DenseVector R
    type Out (RunLayer F) = DenseVector R
    -- trace is (input, weighted-sum)
    newtype Trace (RunLayer F) = DTrace (DenseVector R, DenseVector R)
    forwardT (Full w b) !inp = do
        bv <- newDenseVector (size b)
        bv << inp :<# w
        bv << bv :.+ b
        return $ DTrace (inp,bv)
    output (DTrace (_,!a)) = a
    backward (Full w b) (DTrace (!iv,!bv)) !odelta rate =
        let !d = scale (negate rate) odelta
            !m = iv `outer` d
            -- back-propagated error at input
            !idelta = w #> odelta
            -- update to weights
            ---- for what reason, could this expression: w `add` (iv `outer` d)
            ---- entails a huge space leak? especially, neither 'seq' nor
            ---- 'deepseq' helps a bit. The only workaround is to expand the
            ---- add function, and call SV.force on the result vector, which
            ---- explcitly copy and drop reference to orignal computed result.
            !w'= w `add` m
            -- !w'= let (r,c) = size w
            --          dat1 = flatten (tr' w)
            --          dat2 = flatten (tr' m)
            --      in matrixFromVector ColumnMajor r c $ SV.force $ dat1 `add` dat2
            !b'= b `add` d
            -- !b'= SV.force $ b `add` d
        in (Full w' b', idelta)

instance (Component (RunLayer a),
          Component (RunLayer b),
          Out (RunLayer a) ~ Inp (RunLayer b)
         ) => Component (RunLayer (S a b)) where
    type Run (RunLayer (S a b)) = IO
    type Inp (RunLayer (S a b)) = Inp (RunLayer a)
    type Out (RunLayer (S a b)) = Out (RunLayer b)
    newtype Trace (RunLayer (S a b)) = TTrace (Trace (RunLayer b), Trace (RunLayer a))
    forwardT (Stack a b) !i =
        let !tra = forwardT a i
            !trb = forwardT b (output tra)
        in TTrace (trb, tra)
    output (TTrace !a) = output (fst a)
    backward (Stack a b) (TTrace (!trb,!tra)) !odelta rate =
        let (b', !odelta') = backward b trb odelta  rate
            (a', !idelta ) = backward a tra odelta' rate
        in (Stack a' b', idelta)

newFLayer :: Int                -- number of input values
          -> Int                -- number of neurons (output values)
          -> IO (RunLayer F)    -- new layer
newFLayer m n =
    withSystemRandom . asGenIO $ \gen -> do
        -- we build the weights in column major because in the back-propagation
        -- algo, the computed update to weights is in column major. So it is
        -- good for performance to keep the matrix always in column major.
        w <- buildMatrix (normal 0 0.01 gen) ColumnMajor (m,n)
        b <- return $ konst 1 n
        return $ Full w b

relu, relu' :: R-> R
relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y
