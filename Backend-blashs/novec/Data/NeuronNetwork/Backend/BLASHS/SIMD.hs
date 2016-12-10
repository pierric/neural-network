{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
module Data.NeuronNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import Data.Primitive.SIMD
import Control.Exception

class Num (SIMDPACK a) => SIMDable a where
  type SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()

instance SIMDable Float where
  type SIMDPACK Float = Float
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    go sz v x y
    where
      sz = MV.length v
      go 0 _ _ _ = return ()
      go n z x y = do
        a <- unsafeRead x 0
        b <- unsafeRead y 0
        unsafeWrite z 0 (op a b)
        go (n-1) (unsafeTail z) (unsafeTail x) (unsafeTail y)
