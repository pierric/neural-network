{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
module Data.NeuralNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import qualified Data.Vector.Storable as SV
import Data.Primitive.SIMD
import Control.Exception
import Control.Monad

class Num (SIMDPACK a) => SIMDable a where
  type SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()

instance SIMDable Float where
  type SIMDPACK Float = FloatX4
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    let sv = unsafeCast v :: IOVector FloatX4
        sx = unsafeCast x :: IOVector FloatX4
        sy = unsafeCast y :: IOVector FloatX4
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
    let sv = unsafeCast v :: IOVector FloatX4
        sx = unsafeCast x :: IOVector FloatX4
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

relu, relu' :: FloatX4 -> FloatX4
relu  x = selectVector (compareVector GE x 0) x 0
relu' x = selectVector (compareVector GE 0 x) 0 1

cost' :: FloatX4 -> FloatX4 -> FloatX4
cost' a y = selectVector (compareVector GE a y)
              (selectVector (compareVector GE y 1)
                0
                (a - y))
              (a - y)
