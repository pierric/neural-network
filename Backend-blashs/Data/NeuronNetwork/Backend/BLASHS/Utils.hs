{-# LANGUAGE TypeFamilies, TypeOperators, GADTs #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
module Data.NeuronNetwork.Backend.BLASHS.Utils where

import Blas.Generic.Unsafe
import Blas.Primitive.Types
import qualified Data.Vector.Storable.Mutable as V
import Control.Exception

-- mutable vector type
newtype DenseVector a = DenseVector (V.IOVector a)

-- mutable matrix type
data DenseMatrix a = DenseMatrix Int Int (V.IOVector a)

newDenseVector :: V.Storable a => Int -> IO (DenseVector a)
newDenseVector sz = DenseVector <$> V.new sz

newDenseMatrix :: V.Storable a => Int -> Int -> IO (DenseMatrix a)
newDenseMatrix r c = DenseMatrix r c <$> V.new (r*c)

class Size a where
  type Dim a
  size :: a -> Dim a

instance V.Storable a => Size (DenseVector a) where
  type Dim (DenseVector a) = Int
  size (DenseVector v) = V.length v

instance V.Storable a => Size (DenseMatrix a) where
  type Dim (DenseMatrix a) = (Int,Int)
  size (DenseMatrix r c v) = assert (V.length v >= r * c) $ (r,c)

data Op :: (* -> *) -> * -> * where
  -- vector (by-row) and matrix production
  (:<#) :: DenseVector a -> DenseMatrix a -> Op DenseVector a
  -- matrix and vector (by-column) product
  (:#>) :: DenseMatrix a -> DenseVector a -> Op DenseVector a
  -- vector and vector outer-product
  (:##) :: DenseVector a -> DenseVector a -> Op DenseMatrix a
  -- matrix matrix product
  (:**) :: DenseMatrix a -> DenseMatrix a -> Op DenseMatrix a
  -- pairwise product of vector or matrix
  (:.*) :: c a -> c a -> Op c a
  -- pairwise sum of vector or matrix
  (:.+) :: c a -> c a -> Op c a
  -- scale of vector or matrix
  Scale :: a -> Op c a

class AssignTo c a where
  (<<) :: c a -> Op c a -> IO ()

instance (Numeric a, V.Storable a) => AssignTo DenseVector a where
  (DenseVector v) << (DenseVector x :<# DenseMatrix r c y) =
    V.unsafeWith v (\pv ->
    V.unsafeWith x (\px ->
    V.unsafeWith y (\py ->
      gemv RowMajor Trans r c 1.0 py c px 1 0.0 pv 1)))

  (DenseVector v) << (DenseMatrix r c x :#> DenseVector y) =
    V.unsafeWith v (\pv ->
    V.unsafeWith x (\px ->
    V.unsafeWith y (\py ->
      gemv RowMajor NoTrans r c 1.0 py c px 1 0.0 pv 1)))

  (DenseVector v) << (DenseVector x :.* DenseVector y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y) $
       hadamard (*) v x y

  (DenseVector v) << (DenseVector x :.+ DenseVector y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y) $
       hadamard (+) v x y

  (DenseVector v) << Scale s =
    V.unsafeWith v (\pv -> scal (V.length v) s pv 1)

instance (Numeric a, V.Storable a) => AssignTo DenseMatrix a where
  (DenseMatrix vr vc v) << (DenseVector x :## DenseVector y) =
    let m = V.length x
        n = V.length y
    in assert (m == vr && n == vc) $
       V.unsafeWith v (\pv ->
       V.unsafeWith x (\px ->
       V.unsafeWith y (\py ->
         geru RowMajor m n 1.0 px 1 py 1 pv vc)))
  (DenseMatrix vr vc v) << (DenseMatrix xr xc x :.* DenseMatrix yr yc y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y && vr == xr && vr == yr && vc == xc && vc == yc) $
       hadamard (*) v x y

  (DenseMatrix vr vc v) << (DenseMatrix xr xc x :.* DenseMatrix yr yc y) =
    let sz = V.length v
    in assert (sz == V.length x && sz == V.length y && vr == xr && vr == yr && vc == xc && vc == yc) $
       hadamard (+) v x y

  (DenseMatrix r c v) << Scale s =
    let sz = V.length v
    in assert (sz == r * c) $
       V.unsafeWith v (\pv -> scal sz s pv 1)

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
