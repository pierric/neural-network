{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
module Data.NeuronNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import Data.Primitive.SIMD
import Control.Exception

class Num (SIMDPACK a) => SIMDable a where
  type SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()

instance SIMDable Float where
  type SIMDPACK Float = FloatX4
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz && sz `mod` 4 == 0) $ do
    let sv = unsafeCast v :: IOVector FloatX4
        sx = unsafeCast x :: IOVector FloatX4
        sy = unsafeCast y :: IOVector FloatX4
    go (MV.length sv) sv sx sy
    where
      sz = MV.length v
      go 0 _ _ _ = return ()
      go n z x y = do
        a <- unsafeRead x 0
        b <- unsafeRead y 0
        unsafeWrite z 0 (op a b)
        go (n-1) (unsafeTail z) (unsafeTail x) (unsafeTail y)
