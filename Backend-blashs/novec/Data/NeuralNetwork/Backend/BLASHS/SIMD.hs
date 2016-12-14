{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
module Data.NeuralNetwork.Backend.BLASHS.SIMD where

import Data.Vector.Storable.Mutable as MV
import Control.Exception
import qualified Data.NeuralNetwork as B

class SIMDable a where
  data SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()
  plus  :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  minus :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  times :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a


instance SIMDable Float where
  newtype SIMDPACK Float = F { unF :: Float}
  plus  (F a) (F b) = F (a + b)
  minus (F a) (F b) = F (a - b)
  times (F a) (F b) = F (a * b)
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    go sz v x y
    where
      sz = MV.length v
      go 0 _ _ _ = return ()
      go n z x y = do
        a <- unsafeRead x 0
        b <- unsafeRead y 0
        unsafeWrite z 0 (unF $ op (F a) (F b))
        go (n-1) (unsafeTail z) (unsafeTail x) (unsafeTail y)

  konst = F

  foreach op v x = assert (sz == MV.length x) $ do
    go sz v x
    where
      sz = MV.length v
      go 0 _ _ = return ()
      go n z x = do
        a <- unsafeRead x 0
        unsafeWrite z 0 (unF $ op (F a))
        go (n-1) (unsafeTail z) (unsafeTail x)

relu, relu' :: SIMDPACK Float -> SIMDPACK Float
relu  (F a) = F $ B.relu  a
relu' (F a) = F $ B.relu' a

cost' :: SIMDPACK Float -> SIMDPACK Float -> SIMDPACK Float
cost' (F a) (F b) = F (B.cost' a b)
