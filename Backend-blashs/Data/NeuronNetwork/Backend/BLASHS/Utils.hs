{-# LANGUAGE TypeFamilies, TypeOperators, GADTs #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
module Data.NeuronNetwork.Backend.BLASHS.Utils where

import Blas.Generic.Unsafe
import Blas.Primitive.Types
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as V
import Control.Exception

-- mutable vector type
newtype DenseVector a = DenseVector (V.IOVector a)

-- mutable matrix type
data DenseMatrix a = DenseMatrix Int Int (V.IOVector a)

newDenseVector :: V.Storable a => Int -> IO (DenseVector a)
newDenseVector sz = DenseVector <$> V.new sz

newDenseVectorCopy :: V.Storable a => DenseVector a -> IO (DenseVector a)
newDenseVectorCopy (DenseVector v) = V.clone v >>= return . DenseVector

newDenseVectorConst:: V.Storable a => Int -> a -> IO (DenseVector a)
newDenseVectorConst n v = V.replicate n v >>= return . DenseVector

newDenseMatrix :: V.Storable a => Int -> Int -> IO (DenseMatrix a)
newDenseMatrix r c = DenseMatrix r c <$> V.new (r*c)

newDenseMatrixConst:: V.Storable a => Int -> a -> IO (DenseVector a)
newDenseMatrixConst n v = V.replicate n v >>= return . DenseVector

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

class AssignTo c a where
  (<<=) :: c a -> Op c a -> IO ()
  (<<+) :: c a -> Op c a -> IO ()

instance (Numeric a, V.Storable a) => AssignTo DenseVector a where
  (DenseVector v) <<= (DenseVector x :<# DenseMatrix r c y) =
    V.unsafeWith v (\pv ->
    V.unsafeWith x (\px ->
    V.unsafeWith y (\py ->
      gemv RowMajor Trans r c 1.0 py c px 1 0.0 pv 1)))

  (DenseVector v) <<= (DenseMatrix r c x :#> DenseVector y) =
    V.unsafeWith v (\pv ->
    V.unsafeWith x (\px ->
    V.unsafeWith y (\py ->
      gemv RowMajor NoTrans r c 1.0 py c px 1 0.0 pv 1)))

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

  (DenseVector v) <<+ (DenseVector x :<# DenseMatrix r c y) =
    V.unsafeWith v (\pv ->
    V.unsafeWith x (\px ->
    V.unsafeWith y (\py ->
      gemv RowMajor Trans r c 1.0 py c px 1 1.0 pv 1)))

  (DenseVector v) <<+ (DenseMatrix r c x :#> DenseVector y) =
    V.unsafeWith v (\pv ->
    V.unsafeWith x (\px ->
    V.unsafeWith y (\py ->
      gemv RowMajor NoTrans r c 1.0 py c px 1 1.0 pv 1)))

instance (Numeric a, V.Storable a) => AssignTo DenseMatrix a where
  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.* DenseMatrix yr yc y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y && vr == xr && vr == yr && vc == xc && vc == yc) $
       hadamard (*) v x y

  (DenseMatrix vr vc v) <<= (DenseMatrix xr xc x :.+ DenseMatrix yr yc y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y && vr == xr && vr == yr && vc == xc && vc == yc) $
       hadamard (+) v x y

  (DenseMatrix r c v) <<= Scale s =
    let sz = V.length v
    in assert (sz == r * c) $
       V.unsafeWith v (\pv -> scal sz s pv 1)

  (DenseMatrix vr vc v) <<+ (DenseVector x :## DenseVector y) =
    let m = V.length x
        n = V.length y
    in assert (m == vr && n == vc) $
       V.unsafeWith v (\pv ->
       V.unsafeWith x (\px ->
       V.unsafeWith y (\py ->
         geru RowMajor m n 1.0 px 1 py 1 pv vc)))



hadamard :: (V.Storable a, Num a) => (a -> a -> a) -> V.IOVector a -> V.IOVector a -> V.IOVector a -> IO ()
hadamard op v x y = go 0
  where
    sz = V.length v
    go !i = if (i == sz)
              then return ()
              else do a <- V.unsafeRead x i
                      b <- V.unsafeRead y i
                      V.unsafeWrite v i (op a b)
                      go (i+1)
