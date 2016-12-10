{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
module Data.NeuronNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import qualified Data.Vector.Storable as SV
import Data.Primitive.SIMD
import Control.Exception
import Control.Monad

class Num (SIMDPACK a) => SIMDable a where
  type SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()

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
