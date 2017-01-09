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
{-# LANGUAGE TypeFamilies, TypeOperators, GADTs #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
module Data.NeuralNetwork.Backend.BLASHS.Utils (
  DenseVector(..),
  DenseMatrix(..),
  DenseMatrixArray(..),
  newDenseVector,
  newDenseVectorCopy,
  newDenseVectorConst,
  newDenseVectorByGen,
  newDenseMatrix,
  newDenseMatrixConst,
  newDenseMatrixCopy,
  newDenseMatrixArray,
  Size(..),
  denseVectorToVector,
  denseVectorConcat,
  denseVectorSplit,
  denseVectorCopy,
  denseMatrixArrayAt,
  denseMatrixArrayToVector,
  denseMatrixArrayFromVector,
  v2m, m2v, v2ma, ma2v,
  Op(..), AssignTo(..),
  sumElements, corr2, conv2, pool, unpool, transpose
) where

import Blas.Generic.Unsafe
import Blas.Primitive.Types
import qualified Data.Vector as BV
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V
import Control.Exception
import Control.Monad
import Control.Monad.Trans (MonadIO, liftIO)
import Data.IORef
import Foreign.Marshal.Array (advancePtr)
import Data.NeuralNetwork.Backend.BLASHS.SIMD

-- | mutable vector type
newtype DenseVector a = DenseVector (V.IOVector a)

-- | mutable matrix type (row-major)
data DenseMatrix a = DenseMatrix {-# UNPACK #-}!Int {-# UNPACK #-}!Int {-# UNPACK #-}!(V.IOVector a)

-- | array of DenseMatrix, which are identical in size.
data DenseMatrixArray a = DenseMatrixArray {-# UNPACK #-}!Int {-# UNPACK #-}!Int {-# UNPACK #-}!Int {-# UNPACK #-}!(V.IOVector a)

-- | create a new 'DenseVector'
newDenseVector :: (V.Storable a, MonadIO m) => Int -> m (DenseVector a)
newDenseVector sz = liftIO $ DenseVector <$> V.new sz

-- | create a copy 'DenseVector' from another
newDenseVectorCopy :: (V.Storable a, MonadIO m) => DenseVector a -> m (DenseVector a)
newDenseVectorCopy (DenseVector v) = liftIO $ DenseVector <$> V.clone v

-- | create a new 'DenseVector' of some constant
newDenseVectorConst:: (V.Storable a, MonadIO m) => Int -> a -> m (DenseVector a)
newDenseVectorConst n v = liftIO $ DenseVector <$> V.replicate n v

-- | create a new 'DenseVector' by a random generator
newDenseVectorByGen :: (V.Storable a, MonadIO m) => IO a -> Int -> m (DenseVector a)
newDenseVectorByGen g n = do
  vals <- liftIO $ V.replicateM n g
  return $ DenseVector vals

-- | create a new 'DenseMatrix'
newDenseMatrix :: (V.Storable a, MonadIO m)
               => Int -- ^ number of rows
               -> Int -- ^ number of columns
               -> m (DenseMatrix a)
newDenseMatrix r c = liftIO $ DenseMatrix r c <$> V.new (r*c)

-- | create a new 'DenseMatrix' of some constant
newDenseMatrixConst:: (V.Storable a, MonadIO m) => Int -> Int -> a -> m (DenseMatrix a)
newDenseMatrixConst r c v = liftIO $ DenseMatrix r c <$> V.replicate (r*c) v

-- | create a copy 'DenseMatrix' from another
newDenseMatrixCopy :: (V.Storable a, MonadIO m) => DenseMatrix a -> m (DenseMatrix a)
newDenseMatrixCopy (DenseMatrix r c v) = liftIO $ DenseMatrix r c <$> V.clone v

-- | create a new 'DenseMatrixArray'
newDenseMatrixArray :: (V.Storable a, MonadIO m)
                    => Int -- ^ number of DenseMatrix
                    -> Int -- ^ number of rows
                    -> Int -- ^ number of columns
                    -> m (DenseMatrixArray a)
newDenseMatrixArray n r c = liftIO $ DenseMatrixArray n r c <$> V.new (n*r*c)

-- | get the 'DenseMatrix' from 'DenseMatrixArray' at some position
denseMatrixArrayAt :: V.Storable a => DenseMatrixArray a -> Int -> DenseMatrix a
denseMatrixArrayAt (DenseMatrixArray n r c v) i =
  assert (i >= 0 && i < n) $ let seg = r*c in DenseMatrix r c (V.unsafeSlice (i*seg) seg v)

-- | convert 'DenseMatrixArray' to a vector of 'DenseMatrix' (no copy)
denseMatrixArrayToVector :: V.Storable a => DenseMatrixArray a -> BV.Vector (DenseMatrix a)
denseMatrixArrayToVector (DenseMatrixArray n r c v) =
  let seg = r*c in BV.fromList [DenseMatrix r c (V.unsafeSlice (i*seg) seg v) | i <- [0..n-1]]

-- | convert a vector of 'DenseMatrix' to 'DenseMatrixArray'
-- If all the matrices are orignally placed consecutively in storage, the result
-- is simply a type-cast. Otherwise, a new storage is obtained, and matrices are
-- copied.
denseMatrixArrayFromVector :: (V.Storable a, MonadIO m) => BV.Vector (DenseMatrix a) -> m (DenseMatrixArray a)
denseMatrixArrayFromVector vm = do
  let n = BV.length vm
      DenseMatrix r c (V.MVector _ ptr0) = BV.head vm
  DenseVector raw <- denseVectorConcat (BV.map m2v vm)
  return $ DenseMatrixArray n r c raw

-- | type cast from 'DenseVector' to 'DenseMatrix'
v2m r c (DenseVector v) = DenseMatrix r c v
-- | type cast from 'DenseMatrix' to 'DenseVector'
m2v (DenseMatrix _ _ v) = DenseVector v
-- | type cast from 'DenseVector' to 'DenseMatrixArray'
v2ma n r c (DenseVector v) = assert (V.length v == n*r*c) $ DenseMatrixArray n r c v
-- | type cast from 'DenseMatrixArray' to 'DenseVector'
ma2v (DenseMatrixArray n r c v) = DenseVector v

-- | convert a 'DenseVector' to a vector of elements
denseVectorToVector :: (V.Storable a, MonadIO m) => DenseVector a -> m (BV.Vector a)
denseVectorToVector (DenseVector vs) = liftIO $ SV.unsafeFreeze vs >>= return . BV.convert

-- | concatenate a vector of 'DenseVector's.
-- If all the dense-vectors are orignally placed consecutively in storage, the result
-- is simply a type-cast. Otherwise, a new storage is obtained, and dense-vectors are
-- copied.
denseVectorConcat :: (V.Storable a, MonadIO m)
                  => BV.Vector (DenseVector a) -> m (DenseVector a)
denseVectorConcat vs = do
  let n = BV.length vs
      DenseVector (V.MVector sz0 ptr0) = BV.head vs
  cont <- liftIO $ newIORef True
  size <- liftIO $ newIORef sz0
  forM_ [0..n-2] $ \i -> do
    let DenseVector (V.MVector sz1 ptr1) = vs BV.! i
        DenseVector (V.MVector sz2 ptr2) = vs BV.! (i+1)
    liftIO $ modifyIORef cont (&& (V.getPtr ptr1 `advancePtr` sz1) == V.getPtr ptr2)
    liftIO $ modifyIORef size (+ sz2)
  cont <- liftIO $ readIORef cont
  size <- liftIO $ readIORef size
  if cont
    then do
      return $ DenseVector $ V.unsafeFromForeignPtr0 ptr0 size
    else do
      nvec@(DenseVector rv) <- newDenseVector size
      go rv vs
      return nvec
  where
    go vt vs =
      if BV.null vs
        then assert (V.length vt == 0) $ return ()
        else do
          let DenseVector src = BV.head vs
              (v1, v2) = V.splitAt (V.length src) vt
          liftIO $ V.unsafeCopy v1 src
          go v2 (BV.tail vs)

-- | split a 'DenseVector' into a vector of 'DenseVector's.
denseVectorSplit :: V.Storable a => Int -> Int -> DenseVector a -> BV.Vector (DenseVector a)
denseVectorSplit n c (DenseVector v) = assert (V.length v > n * c) $
  BV.map (\i -> DenseVector (V.unsafeSlice (i*c) c v)) $ BV.enumFromN 0 n

sliceM :: V.Storable a => DenseMatrix a -> (Int, Int) -> DenseVector a
sliceM (DenseMatrix r c d) (x,y) = assert (x>=0 && x<r && y>=0 && y<c) $ DenseVector v
  where
    v = V.unsafeDrop (x*c+y) d

dropV n (DenseVector v) = DenseVector (V.unsafeDrop n v)

copyV (DenseVector v1) (DenseVector v2) len =
  assert (V.length v1 >= len && V.length v2 >= len) $ liftIO $
  V.unsafeCopy (V.unsafeTake len v1) (V.unsafeTake len v2)

denseVectorCopy :: (V.Storable a, MonadIO m) => DenseVector a -> DenseVector a -> m ()
denseVectorCopy t s = assert (size t >= size s) $ copyV t s (size s)

unsafeReadV :: (V.Storable a, MonadIO m) => DenseVector a -> Int -> m a
unsafeReadV (DenseVector v) i = liftIO $ V.unsafeRead v i

unsafeWriteV :: (V.Storable a, MonadIO m) => DenseVector a -> Int -> a -> m ()
unsafeWriteV (DenseVector v) i a = liftIO $ V.unsafeWrite v i a

unsafeReadM :: (V.Storable a, MonadIO m) => DenseMatrix a -> (Int, Int) -> m a
unsafeReadM (DenseMatrix r c v) (i,j) = assert (i < r && j < c) $ liftIO $ V.unsafeRead v (i*c+j)

unsafeWriteM :: (V.Storable a, MonadIO m) => DenseMatrix a -> (Int, Int) -> a -> m ()
unsafeWriteM (DenseMatrix r c v) (i,j) a = assert (i < r && j < c) $ liftIO $ V.unsafeWrite v (i*c+j) a

-- | The Size class provides a interface to tell the dimension of a
-- dense-vector, dense-matrix, or dense-matrix-array.
class Size a where
  type Dim a
  size :: a -> Dim a

instance V.Storable a => Size (DenseVector a) where
  type Dim (DenseVector a) = Int
  size (DenseVector v) = V.length v

instance V.Storable a => Size (DenseMatrix a) where
  type Dim (DenseMatrix a) = (Int,Int)
  size (DenseMatrix r c v) = assert (V.length v >= r * c) $ (r,c)

instance V.Storable a => Size (DenseMatrixArray a) where
  type Dim (DenseMatrixArray a) = (Int,Int,Int)
  size (DenseMatrixArray n r c v) = assert (V.length v >= n * r * c) $ (n,r,c)

infix 4 :<#, :#>, :<>, :##, :.*, :.+
infix 0 <<=, <<+

-- | Operations that abstract the low-level details of blas-hs
data Op :: (* -> *) -> * -> * where
  -- | vector (as-row) and matrix production
  (:<#) :: DenseVector a -> DenseMatrix a -> Op DenseVector a
  -- | matrix and vector (as-column) product
  (:#>) :: DenseMatrix a -> DenseVector a -> Op DenseVector a
  -- | matrix and matrix product.
  -- This is a specially customized matrix matrix product, for the sake of quick
  -- convolution. The 1st matrix is transposed before multiplication, and the
  -- result matrix is stored in column-major mode.
  (:<>) :: DenseMatrix a -> DenseMatrix a -> Op DenseMatrix a
  -- | vector and vector outer-product
  (:##) :: DenseVector a -> DenseVector a -> Op DenseMatrix a
  -- | pairwise product of vector or matrix
  (:.*) :: c a -> c a -> Op c a
  -- | pairwise sum of vector or matrix
  (:.+) :: c a -> c a -> Op c a
  -- | scale of vector or matrix
  Scale :: a -> Op c a
  -- | apply a SIMD-enabled function
  Apply :: (SIMDPACK a -> SIMDPACK a) -> Op c a
  -- | zip with a SIMD-enabled function
  ZipWith :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> c a -> c a -> Op c a
  -- | scale the result of some op.
  -- It is possible to combine scale and many other operations in a single
  -- BLAS call.
  Scale' :: a -> Op c a -> Op c a
  -- | interpret an op to matrix as an op to matrixarray, where each row
  -- becomes a matrix. This Op is only used internally inside this module
  UnsafeM2MA :: Op DenseMatrix a -> Op DenseMatrixArray a

-- | Perform an operation
class AssignTo c a where
  -- | store the result of a Op to the lhs
  (<<=) :: MonadIO m => c a -> Op c a -> m ()
  -- | add the result of a Op to the lhs and store
  (<<+) :: MonadIO m => c a -> Op c a -> m ()

instance (Numeric a, V.Storable a, SIMDable a) => AssignTo DenseVector a where
  (DenseVector v) <<= (DenseVector x :<# DenseMatrix r c y) =
    assert (V.length x == r && V.length v == c) $ liftIO $
      gemv_helper Trans r c 1.0 y c x 0.0 v

  (DenseVector v) <<= (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ liftIO $
      gemv_helper NoTrans r c 1.0 x c y 0.0 v

  (DenseVector v) <<= (DenseVector x :.* DenseVector y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y) $ liftIO $
       hadamard times v x y

  (DenseVector v) <<= (DenseVector x :.+ DenseVector y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y) $ liftIO $
       hadamard plus v x y

  (DenseVector v) <<= Scale s =
    liftIO $ V.unsafeWith v (\pv -> scal (V.length v) s pv 1)

  (DenseVector v) <<= Apply f = liftIO $ foreach f v v

  (DenseVector v) <<= ZipWith f (DenseVector x) (DenseVector y) = liftIO $ hadamard f v x y

  (DenseVector v) <<= Scale' a (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ liftIO $
      gemv_helper NoTrans r c a x c y 0.0 v

  _ <<= _ = error "Unsupported Op [Vector <<=]."

  (DenseVector v) <<+ (DenseVector x :<# DenseMatrix r c y) =
    assert (V.length x == r && V.length v == c) $ liftIO $
      gemv_helper Trans r c 1.0 y c x 1.0 v

  (DenseVector v) <<+ (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ liftIO $
      gemv_helper NoTrans r c 1.0 x c y 1.0 v

  (DenseVector v) <<+ Scale' a (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ liftIO $
      gemv_helper NoTrans r c a x c y 1.0 v

  _ <<+ _ = error "Unsupported Op [Vector <<+]."

instance (Numeric a, V.Storable a, SIMDable a) => AssignTo DenseMatrix a where
  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :<> DenseMatrix yr yc y) =
    assert (xc == yc && vc == xr && vr == yr) $ liftIO $
      gemm_helper Trans NoTrans xr yr xc 1.0 x xc y xc 0.0 v xr

  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.* DenseMatrix yr yc y) =
    assert (vr == xr && vr == yr && vc == xc && vc == yc) $ liftIO $
      hadamard times v x y

  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.+ DenseMatrix yr yc y) =
    assert (vr == xr && vr == yr && vc == xc && vc == yc) $ liftIO $
      hadamard plus v x y

  (DenseMatrix r c v) <<= Scale s =
    let sz = V.length v
    in assert (sz == r * c) $ liftIO $
       V.unsafeWith v (\pv -> scal sz s pv 1)

  (DenseMatrix r c v) <<= Apply f = (DenseVector v) <<= Apply f

  (DenseMatrix vr vc v) <<= Scale' a (DenseMatrix xr xc x :<> DenseMatrix yr yc y) =
    assert (xc == yc && vc == xr && vr == yr) $ liftIO $
      gemm_helper Trans NoTrans xr yr xc a x xc y xc 0.0 v xr

  _ <<= _ = error "Unsupported Op [Matrix <<=]."

  (DenseMatrix vr vc v) <<+ (DenseMatrix xr xc x :<> DenseMatrix yr yc y) =
    assert (xc == yc && vc == xr && vr == yr) $ liftIO $
      gemm_helper Trans NoTrans xr yr xc 1.0 x xc y xc 1.0 v xr

  (DenseMatrix vr vc v) <<+ (DenseVector x :## DenseVector y) =
    let m = V.length x
        n = V.length y
    in assert (m == vr && n == vc) $ liftIO $
       V.unsafeWith v (\pv ->
       V.unsafeWith x (\px ->
       V.unsafeWith y (\py ->
         geru RowMajor m n 1.0 px 1 py 1 pv n)))

  (DenseMatrix vr vc v)  <<+ Scale' a (DenseMatrix xr xc x :<> DenseMatrix yr yc y) =
    assert (xc == yc && vc == xr && vr == yr) $ liftIO $
      gemm_helper Trans NoTrans xr yr xc a x xc y xc 1.0 v xr

  _ <<+ _ = error "Unsupported Op [Matrix <<+]."

instance (Numeric a, V.Storable a, SIMDable a) => AssignTo DenseMatrixArray a where
  ma <<= UnsafeM2MA op = let ma2m (DenseMatrixArray n r c v) = DenseMatrix n (r*c) v
                         in (ma2m ma) <<= op
  ma <<= Scale' r (UnsafeM2MA op) = ma <<= UnsafeM2MA (Scale' r op)
  _ <<= _ = error "Unsupported Op [MatrixArray <<=]."
  ma <<+ UnsafeM2MA op = let ma2m (DenseMatrixArray n r c v) = DenseMatrix n (r*c) v
                         in (ma2m ma) <<+ op
  ma <<+ Scale' r (UnsafeM2MA op) = ma <<+ UnsafeM2MA (Scale' r op)
  _ <<+ _ = error "Unsupported Op [MatrixArray <<+]."

-- | sum up all elements in the 'DenseMatrix'
sumElements :: (V.Storable a, Num a, MonadIO m) => DenseMatrix a -> m a
sumElements (DenseMatrix r c v) = go v (r*c) 0
  where
    go v 0  !s = return s
    go v !n !s = do a <- liftIO $ V.unsafeRead v 0
                    go (V.unsafeTail v) (n-1) (a+s)

-- | 2D correlation.
-- Apply a vector of kernels to a dense-matrix with some zero-padding.
corr2 :: (V.Storable a, Numeric a, MonadIO m)
      => Int                             -- ^ number of 0s padded around
      -> BV.Vector (DenseMatrix a)       -- ^ vector of kernels
      -> DenseMatrix a                   -- ^ matrix to be operated
      -> (Op DenseMatrixArray a -> m b)  -- ^ how to perform the final operation
      -> m b
corr2 p ks m fun = do
  let k0      = BV.head ks
      (kr,kc) = size k0
      (mr,mc) = size m
      u       = mr - kr + 2*p + 1
      v       = mc - kc + 2*p + 1
  zpd <- zero m mr mc p
  wrk <- newDenseMatrix (u*v) (kr*kc)
  fill wrk zpd u v kr kc
  DenseMatrixArray n r c v <- denseMatrixArrayFromVector ks
  fun $ UnsafeM2MA $ wrk :<> DenseMatrix n (r*c) v

-- | 2D convolution.
-- Apply a vector of kernels to a dense-matrix with some zero-padding.
conv2 :: (V.Storable a, Numeric a, MonadIO m)
      => Int                             -- ^ number of 0s padded around
      -> BV.Vector (DenseMatrix a)       -- ^ vector of kernels
      -> DenseMatrix a                   -- ^ matrix to be operated
      -> (Op DenseMatrixArray a -> m b)  -- ^ how to perform the final operation
      -> m b
conv2 p ks m fun = do
  let k0      = BV.head ks
      (kr,kc) = size k0
      (mr,mc) = size m
      u       = mr - kr + 2*p + 1
      v       = mc - kc + 2*p + 1
  zpd <- zero m mr mc p
  wrk <- newDenseMatrix (u*v) (kr*kc)
  fill wrk zpd u v kr kc
  -- copy the kernels, and reverse each.
  let nk      = BV.length ks
  knl@(DenseMatrixArray _ _ _ v) <- newDenseMatrixArray nk kr kc
  forM_ [0..nk-1] $ \i -> do
    let DenseMatrix _ _ d = denseMatrixArrayAt knl i
    let DenseMatrix _ _ s = ks BV.! (nk-1-i)
    liftIO $ V.unsafeCopy d s
  reverseV v
  fun $ UnsafeM2MA $ wrk :<> DenseMatrix nk (kr*kc) v
  where
    reverseV v = let e = V.length v
                     m = e `div` 2
                 in forM_ [0..m] (\i -> liftIO $ V.unsafeSwap v i (e-1-i))

zero m mr mc p = do
  zpd <- newDenseMatrix (mr+2*p) (mc+2*p)
  forM_ [0..mr-1] $ \i -> do
    let t = sliceM zpd (p+i, p)
        s = sliceM m   (  i, 0)
    copyV t s mc
  return zpd

fill wrk@(DenseMatrix _ _ vwrk) m u v kr kc = do
  refv <- liftIO $ newIORef (DenseVector vwrk)
  forM_ [0..u-1] $ \i -> do
    forM_ [0..v-1] $ \j -> do
      forM_ [0..kr-1] $ \k -> do
        t <- liftIO $ readIORef refv
        let s = sliceM m (i+k, j)
        copyV t s kc
        liftIO $ writeIORef refv (dropV kc t)

-- | max-pooling, picking out the maximum element in each stride x stride
-- sub-matrices. Assuming that the original matrix row and column size are
-- both multiple of stride.
pool :: MonadIO m => Int -> DenseMatrix Float -> m (DenseVector Int, DenseMatrix Float)
pool 1 mat = do
  let (r,c) = size mat
  vi <- newDenseVector (r*c)
  return (vi, mat)
pool stride mat = do
  mxi <- newDenseVector (r'*c')
  mxv <- newDenseMatrix r' c'
  forM_ [0..r'-1] $ \i -> do
    forM_ [0..c'-1] $ \j -> do
      (n,v) <- unsafeMaxIndEle mat (i*stride) (j*stride) stride stride
      unsafeWriteV mxi (i*c'+j) n
      unsafeWriteM mxv (i,j)    v
  return (mxi,mxv)
  where
    (r,c) = size mat
    r'    = r `div` stride
    c'    = c `div` stride
    unsafeMaxIndEle mm x y r c = do
      mp <- liftIO $ newIORef 0
      mv <- liftIO $ newIORef (-10000.0)
      forM_ [0..r-1] $ \ i -> do
        forM_ [0..c-1] $ \ j -> do
          v1 <- unsafeReadM mm (x+i, y+j)
          v0 <- liftIO $ readIORef mv
          when (v1 > v0) $ do
            liftIO $ writeIORef mv v1
            liftIO $ writeIORef mp (i*stride+j)
      p <- liftIO $ readIORef mp
      v <- liftIO $ readIORef mv
      return (p, v)

-- | The reverse of max-pooling.
unpool :: MonadIO m => Int -> DenseVector Int -> DenseMatrix Float -> m (DenseMatrix Float)
unpool stride idx mat = do
  mat' <- newDenseMatrix r' c'
  forM_ [0..r-1] $ \i -> do
    forM_ [0..c-1] $ \j -> do
      pos <- unsafeReadV idx (i*c+j)
      val <- unsafeReadM mat (i,j)
      let (oi,oj) = pos `divMod` 2
      unsafeWriteM mat' (i*stride+oi, j*stride+oj) val
  return mat'
  where
    (r,c) = size mat
    (r',c') = (r*stride, c*stride)

-- | transpose a vector of 'DenseMatrixArray'
-- The result is vector of vector of 'DenseMatrix', because the matrices are
-- no longer placed consecutively in storage.
transpose :: (V.Storable a, MonadIO m) => BV.Vector (DenseMatrixArray a) -> m (BV.Vector (BV.Vector (DenseMatrix a)))
transpose vma = do
  let DenseMatrixArray n _ _ _  = BV.head vma
      !vv = BV.map (\i -> BV.map (`denseMatrixArrayAt` i) vma) $ BV.enumFromN 0 n
  return vv

gemv_helper :: (Numeric a, MonadIO m)
            => Transpose
            -> Int -> Int
            -> a
            -> V.IOVector a
            -> Int
            -> V.IOVector a
            -> a
            -> V.IOVector a -> m ()
gemv_helper trans row col alpha x lda y beta v =
  liftIO $
  V.unsafeWith x (\px ->
  V.unsafeWith y (\py ->
  V.unsafeWith v (\pv ->
    gemv RowMajor trans row col alpha px lda py 1 beta pv 1)))

gemm_helper :: (Numeric a, MonadIO m)
            => Transpose
            -> Transpose
            -> Int -> Int -> Int
            -> a
            -> V.IOVector a
            -> Int
            -> V.IOVector a
            -> Int
            -> a
            -> V.IOVector a
            -> Int
            -> m ()
gemm_helper transA transB rowA colB colA alpha x xlda y ylda beta v vlda =
  liftIO $
  V.unsafeWith x (\px ->
  V.unsafeWith y (\py ->
  V.unsafeWith v (\pv -> do
    gemm ColMajor transA transB rowA colB colA alpha px xlda py ylda beta pv vlda)))
