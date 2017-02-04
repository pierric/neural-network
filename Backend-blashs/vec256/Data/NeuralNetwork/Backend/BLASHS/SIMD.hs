{-# LANGUAGE UnboxedTuples, MagicHash #-}
{-# LANGUAGE GHCForeignImportPrim, UnliftedFFITypes #-}
module Data.NeuralNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import qualified Data.Vector.Storable as SV
import Control.Exception
import Control.Monad

import GHC.Prim
import GHC.Base
import GHC.Exts
import GHC.Ptr (Ptr(..))
import Foreign.Storable (Storable(..))

foreign import prim "vfcomp_oge" fcomp_oge :: FloatX8# -> FloatX8# -> Word32X8#
foreign import prim "vselect"    select    :: Word32X8# -> FloatX8# -> FloatX8# -> FloatX8#

data Word32X8 = Word32X8 Word32X8#

data CompareFunc = GE

class SIMDVector v => Comparable v where
  compareVector :: CompareFunc -> v -> v -> Word32X8
  selectVector  :: Word32X8 -> v -> v -> v

instance Comparable (SIMDPACK Float) where
  compareVector GE (FloatX8 x) (FloatX8 y) = Word32X8 (fcomp_oge x y)
  selectVector (Word32X8 s) (FloatX8 x) (FloatX8 y) = FloatX8 (select s x y)

class SIMDable a where
  data SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()
  plus     :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  minus    :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  times    :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a

instance SIMDable Float where
  data SIMDPACK Float = FloatX8 FloatX8#
  plus   (FloatX8 a) (FloatX8 b) = FloatX8 (plusFloatX8#  a b)
  minus  (FloatX8 a) (FloatX8 b) = FloatX8 (minusFloatX8# a b)
  times  (FloatX8 a) (FloatX8 b) = FloatX8 (timesFloatX8# a b)
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    let sv = unsafeCast v :: IOVector (SIMDPACK Float)
        sx = unsafeCast x :: IOVector (SIMDPACK Float)
        sy = unsafeCast y :: IOVector (SIMDPACK Float)
    go (MV.length sv) sv sx sy
    let rm = sz `mod` 8
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
            (vz0,vz1,vz2,vz3,vz4,vz5,vz6,_) = unpackVector (op vx vy)
        forM_ (zip [0..n-1] [vz0,vz1,vz2,vz3,vz4,vz5,vz6]) $ uncurry (unsafeWrite z)

  konst = broadcastVector

  foreach op v x = assert (sz == MV.length x) $ do
    let sv = unsafeCast v :: IOVector (SIMDPACK Float)
        sx = unsafeCast x :: IOVector (SIMDPACK Float)
    go (MV.length sv) sv sx
    let rm = sz `mod` 8
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
            (vz0,vz1,vz2,vz3,vz4,vz5,vz6,_) = unpackVector (op vx)
        forM_ (zip [0..n-1] [vz0,vz1,vz2,vz3,vz4,vz5,vz6]) $ uncurry (unsafeWrite z)

relu, relu' :: SIMDPACK Float -> SIMDPACK Float
relu  x = let v0 = broadcastVector 0
          in selectVector (compareVector GE x v0) x v0
relu' x = let v0 = broadcastVector 0
              v1 = broadcastVector 1
          in selectVector (compareVector GE v0 x) v0 v1

cost' :: SIMDPACK Float -> SIMDPACK Float -> SIMDPACK Float
cost' a y = selectVector (compareVector GE a y)
              (selectVector (compareVector GE y (broadcastVector 1))
                (broadcastVector 0)
                (minus a y))
              (minus a y)

instance Storable (SIMDPACK Float) where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = IO $ \s -> let (# s', r #) = readFloatX8OffAddr# a 0# s in (# s', FloatX8 r #)
    poke (Ptr a) (FloatX8 b) = IO $ \s -> (# writeFloatX8OffAddr# a 0# b s, () #)

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
    type ElemTuple (SIMDPACK Float) = (Float, Float, Float, Float, Float, Float, Float, Float)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 8
    broadcastVector    = broadcastFloatX8
    unsafeInsertVector = unsafeInsertFloatX8
    packVector         = packFloatX8
    unpackVector       = unpackFloatX8

{-# INLINE broadcastFloatX8 #-}
broadcastFloatX8 (F# x) = FloatX8 (broadcastFloatX8# x)

{-# INLINE packFloatX8 #-}
packFloatX8 (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8 ) = FloatX8 (packFloatX8# (# x1, x2, x3, x4, x5, x6, x7, x8 #))

{-# INLINE unpackFloatX8 #-}
unpackFloatX8 (FloatX8 m1) = case unpackFloatX8# m1 of
    (# x1, x2, x3, x4, x5, x6, x7, x8 #) -> (F# x1, F# x2, F# x3, F# x4, F# x5, F# x6, F# x7, F# x8)

{-# INLINE unsafeInsertFloatX8 #-}
unsafeInsertFloatX8 (FloatX8 m1) (F# y) _i@(I# ip) = FloatX8 (insertFloatX8# m1 y (ip -# 0#))
