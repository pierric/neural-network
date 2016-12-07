{-# LANGUAGE TypeFamilies, TypeOperators, GADTs #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
module Data.NeuronNetwork.Backend.BLASHS.Utils where

import Blas.Generic.Unsafe
import Blas.Primitive.Types
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as V
import Control.Exception
import Control.Monad

-- mutable vector type
newtype DenseVector a = DenseVector (V.IOVector a)

-- mutable matrix type
data DenseMatrix a = DenseMatrix Int Int (V.IOVector a)

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
  Apply :: (a -> a) -> Op c a
  -- zip with a function
  ZipWith :: (a -> a -> a) -> c a -> c a -> Op c a
  -- interpret an op to vector as an op to matrix
  UnsafeV2M :: Op DenseVector a -> Op DenseMatrix a
  --
  ScaledOp :: Float -> Op c a -> Op c a

class AssignTo c a where
  (<<=) :: c a -> Op c a -> IO ()
  (<<+) :: c a -> Op c a -> IO ()

instance (Numeric a, V.Storable a) => AssignTo DenseVector a where
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

  (DenseVector v) <<= Apply f = go 0
    where
      sz = V.length v
      go !i | i == sz = return ()
            | otherwise = V.modify v f i >> go (i+1)

  (DenseVector v) <<= ZipWith f (DenseVector x) (DenseVector y) =
    assert (sz1 == sz2 && sz2 == sz3) $ go 0
    where
      sz1 = V.length v
      sz2 = V.length x
      sz3 = V.length y
      go !i | i == sz1 = return ()
            | otherwise = do a <- V.read x i
                             b <- V.read y i
                             V.write v i (f a b)
                             go (i+1)

  (DenseVector v) <<= ScaledOp a (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c a x c y 0.0 v

  (DenseVector v) <<+ (DenseVector x :<# DenseMatrix r c y) =
    assert (V.length x == r && V.length v == c) $ gemv_helper Trans r c 1.0 y c x 1.0 v

  (DenseVector v) <<+ (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c 1.0 x c y 1.0 v

  (DenseVector v) <<+ ScaledOp a (DenseMatrix r c x :#> DenseVector y) =
    assert (V.length y == c && V.length v == r) $ gemv_helper NoTrans r c a x c y 1.0 v

gemv_helper trans row col alpha x lda y beta v =
  V.unsafeWith x (\px ->
  V.unsafeWith y (\py ->
  V.unsafeWith v (\pv ->
    gemv RowMajor trans row col alpha px lda py 1 beta pv 1)))

instance (Numeric a, V.Storable a) => AssignTo DenseMatrix a where
  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.* DenseMatrix yr yc y) =
    assert (vr == xr && vr == yr && vc == xc && vc == yc) $ hadamard (*) v x y

  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.+ DenseMatrix yr yc y) =
    assert (vr == xr && vr == yr && vc == xc && vc == yc) $ hadamard (+) v x y

  (DenseMatrix r c v) <<= Scale s =
    let sz = V.length v
    in assert (sz == r * c) $
       V.unsafeWith v (\pv -> scal sz s pv 1)

  m <<= UnsafeV2M op = (m2v m) <<= op

  m <<= ScaledOp r (UnsafeV2M op) = m <<= UnsafeV2M (ScaledOp r op)

  (DenseMatrix vr vc v) <<+ (DenseVector x :## DenseVector y) =
    let m = V.length x
        n = V.length y
    in assert (m == vr && n == vc) $
       V.unsafeWith v (\pv ->
       V.unsafeWith x (\px ->
       V.unsafeWith y (\py ->
         geru RowMajor m n 1.0 px 1 py 1 pv n)))

  m <<+ UnsafeV2M op = (m2v m) <<+ op

  m <<+ ScaledOp r (UnsafeV2M op) = m <<+ UnsafeV2M (ScaledOp r op)

hadamard :: (V.Storable a, Num a)
         => (a -> a -> a) -> V.IOVector a -> V.IOVector a -> V.IOVector a -> IO ()
hadamard op v x y = assert (V.length x == sz && V.length y == sz) $ go 0
  where
    sz = V.length v
    go !i = if (i == sz)
              then return ()
              else do a <- V.unsafeRead x i
                      b <- V.unsafeRead y i
                      V.unsafeWrite v i (op a b)
                      go (i+1)

corr2 :: (V.Storable a, Numeric a)
      => Int -> DenseMatrix a -> DenseMatrix a -> (Op DenseMatrix a -> IO b) -> IO b
corr2 p k m fun = do
  wrk <- newDenseMatrixConst (u*v) (kr*kc) 0
  fill wrk
  fun $ UnsafeV2M (wrk :#> (m2v k))
  where
    (kr,kc) = size k
    (mr,mc) = size m
    u       = mr - kr + 2*p + 1
    v       = mc - kc + 2*p + 1
    fill wrk = do
      let cpytsk = zip [0..] [gen x y | x<-[-p..mr+p-kr], y<-[-p..mc+p-kc]]
      forM cpytsk $ \ (wr, ts) -> do
        let v = sliceM wrk (wr,0)
        forM ts $ \ (ov, om, len) -> do
          -- putStrLn $ show (wr,ov,om,len)
          copyV (dropV ov v) (sliceM m om) len
    gen x y = [ (ov,(omx,omy), len)
              | omx <- take kr [x..], omx >=0, omx < mr
              , let omy = if y < 0 then 0 else y
              , let len = if y < 0 then kc + y else if y + kc <= mc then kc else mc - y
              , len > 0
              , let ov  = (omx - x)*kc + omy - y ]
