{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
module Data.NeuralNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import Control.Exception
import qualified Data.NeuralNetwork as B

class Num (SIMDPACK a) => SIMDable a where
  type SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()

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

  konst = id

  foreach op v x = assert (sz == MV.length x) $ do
    go sz v x
    where
      sz = MV.length v
      go 0 _ _ = return ()
      go n z x = do
        a <- unsafeRead x 0
        unsafeWrite z 0 (op a)
        go (n-1) (unsafeTail z) (unsafeTail x)

relu, relu' :: Float -> Float
relu  = B.relu
relu' = B.relu'

cost' :: Float -> Float -> Float
cost' = B.cost'
