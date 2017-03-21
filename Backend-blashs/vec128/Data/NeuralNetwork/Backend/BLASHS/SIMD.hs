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

data CompareFunc = GE

class SIMDVector v => Comparable v where
  type Select v
  compareVector :: CompareFunc -> v -> v -> Select v
  selectVector  :: Select v -> v -> v -> v

class (Num a, SIMDVector (SIMDPACK a), Comparable (SIMDPACK a)) => SIMDable a where
  data SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()
  plus     :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  minus    :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  times    :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  divide   :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a

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

-- | SIMD based, RELU and derivative of RELU
relu, relu' :: (SIMDable a,  Comparable (SIMDPACK a)) => SIMDPACK a -> SIMDPACK a
relu  x = let v0 = konst 0
          in selectVector (compareVector GE x v0) x v0
relu' x = let v0 = konst 0
              v1 = konst 1
          in selectVector (compareVector GE v0 x) v0 v1

-- | SIMD based, derivative of error measurement
cost' :: SIMDable a => SIMDPACK a -> SIMDPACK a -> SIMDPACK a
cost' a y = selectVector (compareVector GE a y)
              (selectVector (compareVector GE y (konst 1))
                (konst 0)
                (minus a y))
              (minus a y)

tanh, tanh' :: SIMDable a => SIMDPACK a -> SIMDPACK a
tanh  x = let x2 = times x x
              x3 = times x x2
          in minus x (divide x3 (konst 3))
tanh' x = let a = tanh x
              b = times a a
          in minus (konst 1) b

--------------------------------------------------------------------------------
foreign import prim "vfcomp_oge" fcomp_oge :: FloatX4# -> FloatX4# -> Word32X4#
foreign import prim "vfselect"   fselect   :: Word32X4# -> FloatX4# -> FloatX4# -> FloatX4#

data Word32X4 = Word32X4 Word32X4#

instance Comparable (SIMDPACK Float) where
  type Select (SIMDPACK Float) = Word32X4
  compareVector GE (FloatX4 x) (FloatX4 y) = Word32X4 (fcomp_oge x y)
  selectVector (Word32X4 s) (FloatX4 x) (FloatX4 y) = FloatX4 (fselect s x y)

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

instance Storable (SIMDPACK Float) where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = IO $ \s -> let (# s', r #) = readFloatX4OffAddr# a 0# s in (# s', FloatX4 r #)
    poke (Ptr a) (FloatX4 b) = IO $ \s -> (# writeFloatX4OffAddr# a 0# b s, () #)

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

--------------------------------------------------------------------------------
foreign import prim "vdcomp_oge" dcomp_oge :: DoubleX2# -> DoubleX2# -> Word64X2#
foreign import prim "vdselect"   dselect   :: Word64X2# -> DoubleX2# -> DoubleX2# -> DoubleX2#

data Word64X2 = Word64X2 Word64X2#

instance Comparable (SIMDPACK Double) where
  type Select (SIMDPACK Double) = Word64X2
  compareVector GE (DoubleX2 x) (DoubleX2 y) = Word64X2 (dcomp_oge x y)
  selectVector (Word64X2 s) (DoubleX2 x) (DoubleX2 y) = DoubleX2 (dselect s x y)

instance SIMDable Double where
  data SIMDPACK Double = DoubleX2 DoubleX2#
  plus   (DoubleX2 a) (DoubleX2 b) = DoubleX2 (plusDoubleX2#   a b)
  minus  (DoubleX2 a) (DoubleX2 b) = DoubleX2 (minusDoubleX2#  a b)
  times  (DoubleX2 a) (DoubleX2 b) = DoubleX2 (timesDoubleX2#  a b)
  divide (DoubleX2 a) (DoubleX2 b) = DoubleX2 (divideDoubleX2# a b)
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    let sv = unsafeCast v :: IOVector (SIMDPACK Double)
        sx = unsafeCast x :: IOVector (SIMDPACK Double)
        sy = unsafeCast y :: IOVector (SIMDPACK Double)
    go (MV.length sv) sv sx sy
    let rm = sz `mod` 2
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
      rest 0 z x y = return ()
      rest 1 z x y = do
        x <- MV.unsafeRead x 0
        y <- MV.unsafeRead y 0
        let (vz0, vz1) = unpackVector (op (konst x) (konst y))
        unsafeWrite z 0 vz0

  konst = broadcastVector

  foreach op v x = assert (sz == MV.length x) $ do
    let sv = unsafeCast v :: IOVector (SIMDPACK Double)
        sx = unsafeCast x :: IOVector (SIMDPACK Double)
    go (MV.length sv) sv sx
    let rm = sz `mod` 2
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
      rest 0 z x = return ()
      rest 1 z x = do
        x <- MV.unsafeRead x 0
        let (vz0, vz1) = unpackVector (op (konst x))
        unsafeWrite z 0 vz0

instance Storable (SIMDPACK Double) where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = IO $ \s -> let (# s', r #) = readDoubleX2OffAddr# a 0# s in (# s', DoubleX2 r #)
    poke (Ptr a) (DoubleX2 b) = IO $ \s -> (# writeDoubleX2OffAddr# a 0# b s, () #)

instance SIMDVector (SIMDPACK Double) where
    type Elem (SIMDPACK Double) = Double
    type ElemTuple (SIMDPACK Double) = (Double, Double)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 8
    broadcastVector    = broadcastDoubleX2
    unsafeInsertVector = unsafeInsertDoubleX2
    packVector         = packDoubleX2
    unpackVector       = unpackDoubleX2

{-# INLINE broadcastDoubleX2 #-}
broadcastDoubleX2 (D# x) = DoubleX2 (broadcastDoubleX2# x)

{-# INLINE packDoubleX2 #-}
packDoubleX2 (D# x1, D# x2) = DoubleX2 (packDoubleX2# (# x1, x2 #))

{-# INLINE unpackDoubleX2 #-}
unpackDoubleX2 (DoubleX2 m1) = case unpackDoubleX2# m1 of
    (# x1, x2 #) -> (D# x1, D# x2)

{-# INLINE unsafeInsertDoubleX2 #-}
unsafeInsertDoubleX2 (DoubleX2 m1) (D# y) _i@(I# ip) = DoubleX2 (insertDoubleX2# m1 y (ip -# 0#))
