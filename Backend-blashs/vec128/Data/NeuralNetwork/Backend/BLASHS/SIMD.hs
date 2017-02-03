------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS.SIMD
-- Description :  SIMD based calculations
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module supplies a collection of calculations that
-- could be implemented on top of SIMD.
------------------------------------------------------------
{-# LANGUAGE TypeFamilies, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE UnboxedTuples, MagicHash #-}
{-# LANGUAGE GHCForeignImportPrim, UnliftedFFITypes #-}
module Data.NeuralNetwork.Backend.BLASHS.SIMD (
  compareVector,
  selectVector,
  SIMDable(..),
  cost', relu, relu', tanh, tanh'
) where

import Data.Vector.Storable.Mutable as MV
import qualified Data.Vector.Storable as SV
import Control.Exception
import Control.Monad
import Prelude hiding (tanh)
import GHC.Prim
import GHC.Base
import GHC.Exts
import GHC.Ptr (Ptr(..))
import Foreign.Storable (Storable(..))

foreign import prim "vfcomp_oge" fcomp_oge :: FloatX4# -> FloatX4# -> Word32X4#
foreign import prim "vselect"    select    :: Word32X4# -> FloatX4# -> FloatX4# -> FloatX4#

data Word32X4 = Word32X4 Word32X4#

data CompareFunc = GE

class SIMDVector v => Comparable v where
  compareVector :: CompareFunc -> v -> v -> Word32X4
  selectVector  :: Word32X4 -> v -> v -> v

instance Comparable (SIMDPACK Float) where
  compareVector GE (FloatX4 x) (FloatX4 y) = Word32X4 (fcomp_oge x y)
  selectVector (Word32X4 s) (FloatX4 x) (FloatX4 y) = FloatX4 (select s x y)

class SIMDable a where
  data SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()
  plus     :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  minus    :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  times    :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  divide   :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a

instance SIMDable Float where
  data SIMDPACK Float = FloatX4 FloatX4#
  plus   (FloatX4 a) (FloatX4 b) = FloatX4 (plusFloatX4#  a b)
  minus  (FloatX4 a) (FloatX4 b) = FloatX4 (minusFloatX4# a b)
  times  (FloatX4 a) (FloatX4 b) = FloatX4 (timesFloatX4# a b)
  divide (FloatX4 a) (FloatX4 b) = FloatX4 (divideFloatX4# a b)
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    let sv = unsafeCast v :: IOVector (SIMDPACK Float)
        sx = unsafeCast x :: IOVector (SIMDPACK Float)
        sy = unsafeCast y :: IOVector (SIMDPACK Float)
    go (MV.length sv) sv sx sy
    let rm = sz `mod` 4
        rn = sz - rm
        rv = unsafeDrop rn v
        rx = unsafeDrop rn x
        ry = unsafeDrop rn y
    when (rm /= 0) $ rest rm rv rx ry
    where
      sz = MV.length v
      go 0 _ _ _ = return ()
      go n z x y = do
        a <- unsafeRead x 0
        b <- unsafeRead y 0
        unsafeWrite z 0 (op a b)
        go (n-1) (unsafeTail z) (unsafeTail x) (unsafeTail y)
      rest n z x y = do
        sx <- SV.unsafeFreeze x
        sy <- SV.unsafeFreeze y
        let vx = SV.ifoldl' (\v i a -> unsafeInsertVector v a i) nullVector sx
            vy = SV.ifoldl' (\v i a -> unsafeInsertVector v a i) nullVector sy
            (vz0,vz1,vz2,_) = unpackVector (op vx vy)
        unsafeWrite z 0 vz0
        when (n > 1) $ do
          unsafeWrite z 1 vz1
          when (n > 2) $ do
            unsafeWrite z 2 vz2

  konst = broadcastVector

  foreach op v x = assert (sz == MV.length x) $ do
    let sv = unsafeCast v :: IOVector (SIMDPACK Float)
        sx = unsafeCast x :: IOVector (SIMDPACK Float)
    go (MV.length sv) sv sx
    let rm = sz `mod` 4
        rn = sz - rm
        rv = unsafeDrop rn v
        rx = unsafeDrop rn x
    when (rm /= 0) $ rest rm rv rx
    where
      sz = MV.length v
      go 0 _ _ = return ()
      go n z x = do
        a <- unsafeRead x 0
        unsafeWrite z 0 (op a)
        go (n-1) (unsafeTail z) (unsafeTail x)
      rest n z x = do
        sx <- SV.unsafeFreeze x
        let vx = SV.ifoldl' (\v i a -> unsafeInsertVector v a i) nullVector sx
            (vz0,vz1,vz2,_) = unpackVector (op vx)
        unsafeWrite z 0 vz0
        when (n > 1) $ do
          unsafeWrite z 1 vz1
          when (n > 2) $ do
            unsafeWrite z 2 vz2

-- | SIMD based, RELU and derivative of RELU
relu, relu' :: SIMDPACK Float -> SIMDPACK Float
relu  x = let v0 = broadcastVector 0
          in selectVector (compareVector GE x v0) x v0
relu' x = let v0 = broadcastVector 0
              v1 = broadcastVector 1
          in selectVector (compareVector GE v0 x) v0 v1

-- | SIMD based, derivative of error measurement
cost' :: SIMDPACK Float -> SIMDPACK Float -> SIMDPACK Float
cost' a y = selectVector (compareVector GE a y)
              (selectVector (compareVector GE y (broadcastVector 1))
                (broadcastVector 0)
                (minus a y))
              (minus a y)

tanh, tanh' :: SIMDPACK Float -> SIMDPACK Float
tanh  x = let x2 = times x x
              x3 = times x x2
          in minus x (divide x3 (konst 3))
tanh' x = let a = tanh x
              b = times a a
          in minus (konst 1) b

instance Storable (SIMDPACK Float) where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = IO $ \s -> let (# s', r #) = readFloatX4OffAddr# a 0# s in (# s', FloatX4 r #)
    poke (Ptr a) (FloatX4 b) = IO $ \s -> (# writeFloatX4OffAddr# a 0# b s, () #)

class SIMDVector v where
    -- | Type of the elements in the vector
    type Elem v
    -- | Type used to pack or unpack the vector
    type ElemTuple v
    -- | Vector with all elements initialized to zero.
    nullVector       :: v
    -- | Number of components (scalar elements) in the vector. The argument is not evaluated.
    vectorSize       :: v -> Int
    -- | Size of each (scalar) element in the vector in bytes. The argument is not evaluated.
    elementSize      :: v -> Int
    -- | Broadcast a scalar to all elements of a vector.
    broadcastVector  :: Elem v -> v
    -- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range an exception is thrown.
    insertVector     :: v -> Elem v -> Int -> v
    insertVector v e i | i < 0            = error $ "insertVector: negative argument: " ++ show i
                       | i < vectorSize v = unsafeInsertVector v e i
                       | otherwise        = error $ "insertVector: argument too large: " ++ show i
    -- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range the behavior is undefined.
    unsafeInsertVector     :: v -> Elem v -> Int -> v
    -- | Pack some elements to a vector.
    packVector       :: ElemTuple v -> v
    -- | Unpack a vector.
    unpackVector     :: v -> ElemTuple v

instance SIMDVector (SIMDPACK Float) where
    type Elem (SIMDPACK Float) = Float
    type ElemTuple (SIMDPACK Float) = (Float, Float, Float, Float)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 4
    broadcastVector    = broadcastFloatX4
    unsafeInsertVector = unsafeInsertFloatX4
    packVector         = packFloatX4
    unpackVector       = unpackFloatX4

{-# INLINE broadcastFloatX4 #-}
broadcastFloatX4 (F# x) = FloatX4 (broadcastFloatX4# x)

{-# INLINE packFloatX4 #-}
packFloatX4 (F# x1, F# x2, F# x3, F# x4) = FloatX4 (packFloatX4# (# x1, x2, x3, x4 #))

{-# INLINE unpackFloatX4 #-}
unpackFloatX4 (FloatX4 m1) = case unpackFloatX4# m1 of
    (# x1, x2, x3, x4 #) -> (F# x1, F# x2, F# x3, F# x4)

{-# INLINE unsafeInsertFloatX4 #-}
unsafeInsertFloatX4 (FloatX4 m1) (F# y) _i@(I# ip) = FloatX4 (insertFloatX4# m1 y (ip -# 0#))
