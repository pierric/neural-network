{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts, GADTs #-}
module Data.NeuronNetwork.Backend.HMatrix.Layers where

import Numeric.LinearAlgebra hiding (R, C)
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SVM
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2, forM_, when)
import GHC.Float
import Data.STRef
import Control.DeepSeq
import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.HMatrix.Utils

type R = Float

data F
data C
data A
data S a b
data RunLayer :: * -> * where
  -- Densely connected layer
  -- input:   vector of size m
  -- output:  vector of size n
  -- weights: matrix of size m x n
  -- biases:  vector of size n
  Full :: !(Matrix R) -> !(Vector R) -> (R -> R, R -> R) -> RunLayer F
  -- convolutional layer
  -- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
  -- output: channels of 2D floats, of the same size (c x d), # of output channels: n
  --         where c = a + 2*padding + 1 - s
  --               d = b + 2*padding + 1 - t
  -- feature:  matrix of (s x t), # of features: m x n
  -- padding:  number of 0s padded at each side of channel
  -- biases:   bias for each output, # of biases: n
  Conv  :: !(V.Vector (V.Vector (Matrix R))) -> !(V.Vector R) -> Int -> RunLayer C
  -- Reshape from channels of matrix to a single vector
  -- input:  m channels of 2D matrices
  --         assuming that all matrices are of the same size a x b
  -- output: 1D vector of the concatenation of all input channels
  --         its size: m x a x b
  As1D  :: RunLayer A
  -- stacking two components a and b
  -- the output of a should matches the input of b
  Stack :: !(RunLayer a) -> !(RunLayer b) -> RunLayer (S a b)

instance Component (RunLayer F) where
    type Inp (RunLayer F) = Vector R
    type Out (RunLayer F) = Vector R
    -- trace is (input, weighted-sum, output)
    newtype Trace (RunLayer F) = DTrace (Vector R, Vector R, Vector R)
    forwardT (Full w b (af,_)) !inp =
        let !bv = (inp <# w) `add` b
            !av = cmap af bv
        in DTrace (inp,bv,av)
    output (DTrace (_,_,!a)) = a
    backward l (DTrace (!iv,!bv,_)) !odelta rate =
        let Full w b ac@(_,af') = l
            -- back-propagated error before activation
            udelta = odelta `hadamard` cmap af' bv
            !d = scale (negate rate) udelta
            !m = iv `outer` d
            -- back-propagated error at input
            !idelta = w #> udelta
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
        in (Full w' b' ac, idelta)

instance Component (RunLayer C) where
    type Inp (RunLayer C) = V.Vector (Matrix R)
    type Out (RunLayer C) = V.Vector (Matrix R)
    -- trace is (input, convoluted output)
    newtype Trace (RunLayer C) = CTrace (Inp (RunLayer C), V.Vector (Matrix R))
    forwardT (Conv fs bs p) !inp =
        let !ov = parallel $ V.zipWith feature
                               (tr fs) -- feature matrix indexed majorly by each output
                               bs      -- biases by each output
        in CTrace (inp,ov)
      where
        !osize = let (x,y) = size (V.head inp)
                     (u,v) = size (V.head $ V.head fs)
                 in (x+2*p-u+1, y+2*p-v+1)
        -- transpose the features matrix
        tr :: V.Vector (V.Vector a) -> V.Vector (V.Vector a)
        tr uv = let n   = V.length (V.head uv)
                    !vu = V.map (\i -> V.map (V.! i) uv) $ V.enumFromN 0 n
                in vu
        feature :: V.Vector (Matrix R) -> R -> Matrix R
        feature f b = V.foldl1' add (V.zipWith (layerCorr2 p) f inp) `add` konst b osize
    output (CTrace (_,a)) = a
    backward l (CTrace (!iv,!av)) !odelta rate =
      let Conv fs bs p = l
          -- update to the feature matrix
          m :: V.Vector (V.Vector (Matrix R))
          !m = parallel $ V.zipWith (\flts chn ->
                            -- chn:  a single input channel
                            -- flts: all features used for chn
                            V.zipWith (\f d ->
                              let upd = scale (negate rate) (layerCorr2 p chn d)
                              in f `add` upd
                            ) flts odelta
                          ) fs iv
          -- update to the biases
          b :: V.Vector R
          !b = V.zipWith (\b d -> b + (negate rate) * sumElements d) bs odelta
          -- back-propagated error at input
          idelta :: V.Vector (Matrix R)
          !idelta = V.map (\f -> V.foldl1' add $ V.zipWith (layerConv2 p) f odelta) fs
      in --trace ("CL:" ++ show odelta)
         (Conv m b p, idelta)

instance Component (RunLayer A) where
  type Inp (RunLayer A) = V.Vector (Matrix R)
  type Out (RunLayer A) = Vector R
  -- trace keeps information of (m, axb, b, output)
  newtype Trace (RunLayer A) = ReshapeTrace (Int, Int, Int, Vector R)
  forwardT _ !inp =
    let !b = V.length inp
        (!r,!c) = size (V.head inp)
        !o = V.foldr' (\x y -> flatten x SV.++ y) SV.empty inp
    in ReshapeTrace (b, r*c, c, o)
  output (ReshapeTrace (_,_,_,a)) = a
  backward a (ReshapeTrace (b,n,c,_)) !odelta _ =
    let !idelta = V.fromList $ map (reshape c) $ takesV (replicate b n) odelta
    in (a, idelta)

instance (Component (RunLayer a),
          Component (RunLayer b),
          Out (RunLayer a) ~ Inp (RunLayer b)
         ) => Component (RunLayer (S a b)) where
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

newDLayer :: (Int, Int)         -- number of input channels, number of neurons (output channels)
          -> (R->R, R->R)       -- activate function and its derivative
          -> IO (RunLayer F)    -- new layer
newDLayer sz@(_,n) (af, af') =
    withSystemRandom . asGenIO $ \gen -> do
        -- we build the weights in column major because in the back-propagation
        -- algo, the computed update to weights is in column major. So it is
        -- good for performance to keep the matrix always in column major.
        w <- buildMatrix (normal 0 0.01 gen) ColumnMajor sz
        b <- return $ konst 1 n
        return $ Full w b (af, af')

newCLayer :: Int                -- number of input channels
          -> Int                -- number of output channels
          -> Int                -- size of each feature
          -> Int                -- size of padding
          -> IO (RunLayer C)    -- new layer
newCLayer inpsize outsize sfilter npadding =
  withSystemRandom . asGenIO $ \gen -> do
      fs <- V.replicateM inpsize $ V.replicateM outsize $
              buildMatrix (truncNormal 0 0.1 gen) RowMajor (sfilter, sfilter)
      bs <- return $ V.replicate outsize 0.1
      return $ Conv fs bs npadding
  where
    truncNormal m s g = do
      x <- standard g
      if x >= 2.0 || x <= -2.0
        then truncNormal m s g
        else return $! m + s * x

buildMatrix g order (nr, nc) = do
  vals <- SV.replicateM (nr*nc) (double2Float <$> g)
  return $ matrixFromVector order nr nc vals

layerCorr2 :: Int -> Matrix R -> Matrix R -> Matrix R
layerCorr2 p k m = c_corr2d_s k padded
  where
    padded = zeroPadded p m
    (w,_)  = size k

layerConv2 :: Int -> Matrix R -> Matrix R -> Matrix R
layerConv2 p k m = c_conv2d_s k padded
  where
    padded = zeroPadded p m
    (w,_)  = size k

relu, relu' :: R-> R
relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

-- max pool, picking out the maximum element
-- in each stride x stride sub-matrices.
-- assuming that the original matrix row and column size are
-- both multiple of stride
pool :: Int -> Matrix Float -> (Vector Int, Matrix Float)
pool 1 mat = let (r,c) = size mat in (SV.replicate (r*c) 0, mat)
-- pool 2 mat | orderOf mat == RowMajor = c_max_pool2_f mat
pool stride mat = runST $ do
  ori <- unsafeThawMatrix mat
  mxv <- newUndefinedMatrix RowMajor r' c'
  mxi <- newUndefinedVector (r'*c')
  forM_ [0..r'-1] $ \i -> do
    forM_ [0..c'-1] $ \j -> do
      (n,v) <- unsafeMaxIndEle ori (i*stride) (j*stride) stride stride
      unsafeWriteVector mxi (i*c'+j) n
      unsafeWriteMatrix mxv i j v
  a <- unsafeFreezeVector mxi
  b <- unsafeFreezeMatrix mxv
  return (a,b)
  where
    (r,c) = size mat
    r'    = r `div` stride
    c'    = c `div` stride
    unsafeMaxIndEle mm x y r c = do
      mp <- newSTRef 0
      mv <- newSTRef (-10000.0)
      forM_ [0..r-1] $ \ i -> do
        forM_ [0..c-1] $ \ j -> do
          v1 <- unsafeReadMatrix mm (x+i) (y+j)
          v0 <- readSTRef mv
          when (v1 > v0) $ do
            writeSTRef mv v1
            writeSTRef mp (i*2+j)
      p <- readSTRef mp
      v <- readSTRef mv
      return (p, v)

-- the reverse of max pool.
-- assuming idx and mat are of the same size
unpool :: Int -> Vector Int -> Matrix Float -> Matrix Float
unpool stride idx mat = runSTMatrix $ do
  mat' <- newMatrix' 0 r' c'
  forM_ [0..r-1] $ \i -> do
    forM_ [0..c-1] $ \j -> do
      let pos     = idx SV.! (i*c+j)
      let (oi,oj) = pos `divMod` 2
      let val     = mat `atIndex` (i,j)
      unsafeWriteMatrix mat' (i*stride+oi) (j*stride+oj) val
  return mat'
  where
    (r,c) = size mat
    (r',c') = (r*stride, c*stride)

-- a slightly faster way to pading the matrix
-- camparing to fromBlocks provided by hmatrix.
zeroPadded :: Int -> Matrix Float -> Matrix Float
zeroPadded p mat = runSTMatrix $ do
  mat' <- newMatrix' 0 r' c'
  setMatrix mat' p p mat
  return mat'
  where
    (r,c) = size mat
    (r',c') = (r+2*p,c+2*p)

-- a slightly faster version of newMatrix, which based
-- directly on lower level Vector.Storable creation.
newMatrix' :: SVM.Storable t => t -> Int -> Int -> ST s (STMatrix s t)
newMatrix' v r c = do
  vec <- SVM.replicate (r*c) v
  vec <- SV.unsafeFreeze vec
  unsafeThawMatrix $ reshape c vec
