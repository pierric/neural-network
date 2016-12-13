{-# LANGUAGE TypeFamilies, TypeOperators, GADTs #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
module Data.NeuralNetwork.Backend.BLASHS.Utils where

import Blas.Generic.Unsafe
import Blas.Primitive.Types
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as V
import Control.Exception
import Control.Monad
import Data.IORef
import Data.NeuralNetwork.Backend.BLASHS.SIMD

-- mutable vector type
newtype DenseVector a = DenseVector (V.IOVector a)

-- mutable matrix type
data DenseMatrix a = DenseMatrix {-# UNPACK #-}!Int {-# UNPACK #-}!Int {-# UNPACK #-}!(V.IOVector a)

newDenseVector :: V.Storable a => Int -> IO (DenseVector a)
newDenseVector sz = DenseVector <$> V.unsafeNew sz

newDenseVectorCopy :: V.Storable a => DenseVector a -> IO (DenseVector a)
newDenseVectorCopy (DenseVector v) = V.clone v >>= return . DenseVector

newDenseVectorConst:: V.Storable a => Int -> a -> IO (DenseVector a)
newDenseVectorConst n v = V.replicate n v >>= return . DenseVector

newDenseMatrix :: V.Storable a => Int -> Int -> IO (DenseMatrix a)
newDenseMatrix r c = DenseMatrix r c <$> V.unsafeNew (r*c)

newDenseMatrixConst:: V.Storable a => Int -> Int -> a -> IO (DenseMatrix a)
newDenseMatrixConst r c v = V.replicate (r*c) v >>= return . DenseMatrix r c

newDenseMatrixByGen :: IO Float -> Int -> Int -> IO (DenseMatrix Float)
newDenseMatrixByGen g nr nc = do
  vals <- V.replicateM (nr*nc) g
  return $ DenseMatrix nr nc vals

newDenseMatrixCopy :: V.Storable a => DenseMatrix a -> IO (DenseMatrix a)
newDenseMatrixCopy (DenseMatrix r c v) = V.clone v >>= return . DenseMatrix r c

v2m r c (DenseVector v) = DenseMatrix r c v
m2v (DenseMatrix _ _ v) = DenseVector v

toListV (DenseVector vs) = SV.unsafeFreeze vs >>= return . SV.toList

concatV :: V.Storable a => [DenseVector a] -> IO (DenseVector a)
concatV vs = do
  let sz = sum $ map (\(DenseVector v) -> V.length v) vs
  rv <- V.unsafeNew sz
  go rv vs
  return $ DenseVector rv
  where
    go vt [] = assert (V.length vt == 0) $ return ()
    go vt (DenseVector vs:vss) = assert (V.length vt >= V.length vs) $ do
      let (v1, v2) = V.splitAt (V.length vs) vt
      V.unsafeCopy v1 vs
      go v2 vss

splitV :: V.Storable a => Int -> Int -> DenseVector a -> [DenseVector a]
splitV n c (DenseVector v) = assert (V.length v > n * c) $
  [DenseVector (V.unsafeSlice (i*c) c v) | i <- [0..n-1]]

sliceM :: V.Storable a => DenseMatrix a -> (Int, Int) -> DenseVector a
sliceM (DenseMatrix r c d) (x,y) = assert (x>=0 && x<r && y>=0 && y<c) $ DenseVector v
  where
    v = V.unsafeDrop (x*c+y) d

unsafeReadV :: V.Storable a => DenseVector a -> Int -> IO a
unsafeReadV (DenseVector v) i = V.unsafeRead v i

unsafeWriteV :: V.Storable a => DenseVector a -> Int -> a -> IO ()
unsafeWriteV (DenseVector v) i a = V.unsafeWrite v i a

unsafeReadM :: V.Storable a => DenseMatrix a -> (Int, Int) -> IO a
unsafeReadM (DenseMatrix r c v) (i,j) = assert (i < r && j < c) $ V.unsafeRead v (i*c+j)

unsafeWriteM :: V.Storable a => DenseMatrix a -> (Int, Int) -> a -> IO ()
unsafeWriteM (DenseMatrix r c v) (i,j) a = assert (i < r && j < c) $ V.unsafeWrite v (i*c+j) a

dropV n (DenseVector v) = DenseVector (V.unsafeDrop n v)

copyV (DenseVector v1) (DenseVector v2) len =
  assert (V.length v1 >= len && V.length v2 >= len) $
  V.unsafeCopy (V.unsafeTake len v1) (V.unsafeTake len v2)

class Size a where
  type Dim a
  size :: a -> Dim a

instance V.Storable a => Size (DenseVector a) where
  type Dim (DenseVector a) = Int
  size (DenseVector v) = V.length v

instance V.Storable a => Size (DenseMatrix a) where
  type Dim (DenseMatrix a) = (Int,Int)
  size (DenseMatrix r c v) = assert (V.length v >= r * c) $ (r,c)

infix 4 :<#, :#>, :##, :.*, :.+
infix 0 <<=, <<+

data Op :: (* -> *) -> * -> * where
  -- vector (by-row) and matrix production
  (:<#) :: DenseVector a -> DenseMatrix a -> Op DenseVector a
  -- matrix and vector (by-column) product
  (:#>) :: DenseMatrix a -> DenseVector a -> Op DenseVector a
  -- vector and vector outer-product
  (:##) :: DenseVector a -> DenseVector a -> Op DenseMatrix a
  -- pairwise product of vector or matrix
  (:.*) :: c a -> c a -> Op c a
  -- pairwise sum of vector or matrix
  (:.+) :: c a -> c a -> Op c a
  -- scale of vector or matrix
  Scale :: a -> Op c a
  -- apply a function
  Apply :: (SIMDPACK a -> SIMDPACK a) -> Op c a
  -- zip with a function
  ZipWith :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> c a -> c a -> Op c a
  -- interpret an op to vector as an op to matrix
  UnsafeV2M :: Op DenseVector a -> Op DenseMatrix a
  -- scale the result of some op
  -- especially used in convolution, where :#> is followed by scale.
  Scale' :: a -> Op c a -> Op c a

class AssignTo c a where
  (<<=) :: c a -> Op c a -> IO ()
  (<<+) :: c a -> Op c a -> IO ()

instance (Numeric a, V.Storable a, SIMDable a) => AssignTo DenseVector a where
  (DenseVector v) <<= (DenseVector x :<# DenseMatrix r c y) =
    assert (V.length x == r && V.length v == c) $ gemv_helper Trans r c 1.0 y c x 0.0 v

  (DenseVector v) <<= (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c 1.0 x c y 0.0 v

  (DenseVector v) <<= (DenseVector x :.* DenseVector y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y) $
       hadamard (*) v x y

  (DenseVector v) <<= (DenseVector x :.+ DenseVector y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y) $
       hadamard (+) v x y

  (DenseVector v) <<= Scale s =
    V.unsafeWith v (\pv -> scal (V.length v) s pv 1)

  (DenseVector v) <<= Apply f = foreach f v v

  (DenseVector v) <<= ZipWith f (DenseVector x) (DenseVector y) = hadamard f v x y

  (DenseVector v) <<= Scale' a (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c a x c y 0.0 v

  _ <<= _ = error "Unsupported Op [Vector <<=]."

  (DenseVector v) <<+ (DenseVector x :<# DenseMatrix r c y) =
    assert (V.length x == r && V.length v == c) $ gemv_helper Trans r c 1.0 y c x 1.0 v

  (DenseVector v) <<+ (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c 1.0 x c y 1.0 v

  (DenseVector v) <<+ Scale' a (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c a x c y 1.0 v

  _ <<+ _ = error "Unsupported Op [Vector <<+]."

gemv_helper :: Numeric a
            => Transpose
            -> Int -> Int
            -> a
            -> V.IOVector a
            -> Int
            -> V.IOVector a
            -> a
            -> V.IOVector a -> IO ()
gemv_helper trans row col alpha x lda y beta v =
  V.unsafeWith x (\px ->
  V.unsafeWith y (\py ->
  V.unsafeWith v (\pv ->
    gemv RowMajor trans row col alpha px lda py 1 beta pv 1)))

instance (Numeric a, V.Storable a, SIMDable a) => AssignTo DenseMatrix a where
  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.* DenseMatrix yr yc y) =
    assert (vr == xr && vr == yr && vc == xc && vc == yc) $ hadamard (*) v x y

  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.+ DenseMatrix yr yc y) =
    assert (vr == xr && vr == yr && vc == xc && vc == yc) $ hadamard (+) v x y

  (DenseMatrix r c v) <<= Scale s =
    let sz = V.length v
    in assert (sz == r * c) $
       V.unsafeWith v (\pv -> scal sz s pv 1)

  (DenseMatrix r c v) <<= Apply f = (DenseVector v) <<= Apply f

  m <<= UnsafeV2M op = (m2v m) <<= op

  m <<= Scale' r (UnsafeV2M op) = m <<= UnsafeV2M (Scale' r op)

  _ <<= _ = error "Unsupported Op [Matrix <<=]."

  (DenseMatrix vr vc v) <<+ (DenseVector x :## DenseVector y) =
    let m = V.length x
        n = V.length y
    in assert (m == vr && n == vc) $
       V.unsafeWith v (\pv ->
       V.unsafeWith x (\px ->
       V.unsafeWith y (\py ->
         geru RowMajor m n 1.0 px 1 py 1 pv n)))

  m <<+ UnsafeV2M op = (m2v m) <<+ op

  m <<+ Scale' r (UnsafeV2M op) = m <<+ UnsafeV2M (Scale' r op)

  _ <<+ _ = error "Unsupported Op [Matrix <<+]."

-- hadamard :: (V.Storable a, Num a)
--          => (a -> a -> a) -> V.IOVector a -> V.IOVector a -> V.IOVector a -> IO ()
-- hadamard op v x y = assert (V.length x == sz && V.length y == sz) $ go 0
--   where
--     sz = V.length v
--     go !i = if (i == sz)
--               then return ()
--               else do a <- V.unsafeRead x i
--                       b <- V.unsafeRead y i
--                       V.unsafeWrite v i (op a b)
--                       go (i+1)

sumElements :: (V.Storable a, Num a) => DenseMatrix a -> IO a
sumElements (DenseMatrix r c v) = go v (r*c) 0
  where
    go v 0  !s = return s
    go v !n !s = do a <- V.unsafeRead v 0
                    go (V.unsafeTail v) (n-1) (a+s)

corr2 :: (V.Storable a, Numeric a)
      => Int -> DenseMatrix a -> DenseMatrix a -> (Op DenseMatrix a -> IO b) -> IO b
corr2 p k m fun = do
  let (kr,kc) = size k
      (mr,mc) = size m
      u       = mr - kr + 2*p + 1
      v       = mc - kc + 2*p + 1
  zpd <- zero m mr mc p
  wrk <- newDenseMatrixConst (u*v) (kr*kc) 0
  fill wrk zpd u v kr kc
  fun $ UnsafeV2M (wrk :#> (m2v k))

conv2 :: (V.Storable a, Numeric a)
      => Int -> DenseMatrix a -> DenseMatrix a -> (Op DenseMatrix a -> IO b) -> IO b
conv2 p k m fun = do
  let (kr,kc) = size k
      (mr,mc) = size m
      u       = mr - kr + 2*p + 1
      v       = mc - kc + 2*p + 1
  zpd <- zero m mr mc p
  wrk <- newDenseMatrixConst (u*v) (kr*kc) 0
  fill wrk zpd u v kr kc
  k' <- case k of DenseMatrix _ _ v -> SV.unsafeFreeze v
  k' <- SV.unsafeThaw (SV.reverse k')
  fun $ UnsafeV2M (wrk :#> (DenseVector k'))

zero m mr mc p = do
  zpd <- newDenseMatrixConst (mr+2*p) (mc+2*p) 0
  forM_ [0..mr-1] $ \i -> do
    let t = sliceM zpd (p+i, p)
        s = sliceM m   (  i, 0)
    copyV t s mc
  return zpd

fill wrk@(DenseMatrix _ _ vwrk) m u v kr kc = do
  refv <- newIORef (DenseVector vwrk)
  forM_ [0..u-1] $ \i -> do
    forM_ [0..v-1] $ \j -> do
      forM_ [0..kr-1] $ \k -> do
        t <- readIORef refv
        let s = sliceM m (i+k, j)
        copyV t s kc
        writeIORef refv (dropV kc t)

-- fill wrk p kr kc mr mc m = do
--   let cpytsk = zip [0..] [gen x y | x<-[-p..mr+p-kr], y<-[-p..mc+p-kc]]
--   forM_ cpytsk $ \ (wr, ts) -> do
--     let v = sliceM wrk (wr,0)
--     forM_ ts $ \ (ov, om, len) -> do
--       -- putStrLn $ show (wr,ov,om,len)
--       copyV (dropV ov v) (sliceM m om) len
--   where
--     gen x y = [ (ov,(omx,omy), len)
--               | omx <- take kr [x..], omx >=0, omx < mr
--               , let omy = if y < 0 then 0 else y
--               , let len = if y < 0 then kc + y else if y + kc <= mc then kc else mc - y
--               , len > 0
--               , let ov  = (omx - x)*kc + omy - y ]

-- max pool, picking out the maximum element
-- in each stride x stride sub-matrices.
-- assuming that the original matrix row and column size are
-- both multiple of stride
pool :: Int -> DenseMatrix Float -> IO (DenseVector Int, DenseMatrix Float)
pool 1 mat = do
  let (r,c) = size mat
  vi <- newDenseVectorConst (r*c) 0
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
      mp <- newIORef 0
      mv <- newIORef (-10000.0)
      forM_ [0..r-1] $ \ i -> do
        forM_ [0..c-1] $ \ j -> do
          v1 <- unsafeReadM mm (x+i, y+j)
          v0 <- readIORef mv
          when (v1 > v0) $ do
            writeIORef mv v1
            writeIORef mp (i*stride+j)
      p <- readIORef mp
      v <- readIORef mv
      return (p, v)

-- the reverse of max pool.
-- assuming idx and mat are of the same size
unpool :: Int -> DenseVector Int -> DenseMatrix Float -> IO (DenseMatrix Float)
unpool stride idx mat = do
  mat' <- newDenseMatrixConst r' c' 0
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
