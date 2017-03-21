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
{-# LANGUAGE StandaloneDeriving, GeneralizedNewtypeDeriving #-}
module Data.NeuralNetwork.Backend.BLASHS.SIMD (
  SIMDable(..),
  cost', relu, relu', tanh, tanh'
)  where

import Data.Vector.Storable.Mutable as MV
import Control.Exception
import qualified Data.NeuralNetwork as B
import qualified Prelude as Prelude (tanh)
import Prelude hiding (tanh)

class SIMDable a where
  data SIMDPACK a
  hadamard :: (SIMDPACK a -> SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IOVector a -> IO ()
  konst    :: a -> SIMDPACK a
  foreach  :: (SIMDPACK a -> SIMDPACK a) -> IOVector a -> IOVector a -> IO ()
  plus  :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  minus :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  times :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a
  divide   :: SIMDPACK a -> SIMDPACK a -> SIMDPACK a

instance SIMDable Float where
  newtype SIMDPACK Float = F { unF :: Float}
  plus  (F a) (F b) = F (a + b)
  minus (F a) (F b) = F (a - b)
  times (F a) (F b) = F (a * b)
  divide (F a) (F b) = F (a / b)
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

instance SIMDable Double where
  newtype SIMDPACK Double = D { unD :: Double}
  plus  (D a) (D b) = D (a + b)
  minus (D a) (D b) = D (a - b)
  times (D a) (D b) = D (a * b)
  divide (D a) (D b) = D (a / b)
  hadamard op v x y = assert (MV.length x == sz && MV.length y == sz) $ do
    go sz v x y
    where
      sz = MV.length v
      go 0 _ _ _ = return ()
      go n z x y = do
        a <- unsafeRead x 0
        b <- unsafeRead y 0
        unsafeWrite z 0 (unD $ op (D a) (D b))
        go (n-1) (unsafeTail z) (unsafeTail x) (unsafeTail y)

  konst = D

  foreach op v x = assert (sz == MV.length x) $ do
    go sz v x
    where
      sz = MV.length v
      go 0 _ _ = return ()
      go n z x = do
        a <- unsafeRead x 0
        unsafeWrite z 0 (unD $ op (D a))
        go (n-1) (unsafeTail z) (unsafeTail x)

deriving instance Eq  (SIMDPACK Float)
deriving instance Num (SIMDPACK Float)
deriving instance Ord (SIMDPACK Float)
deriving instance Fractional (SIMDPACK Float)
deriving instance Floating   (SIMDPACK Float)
deriving instance Eq  (SIMDPACK Double)
deriving instance Num (SIMDPACK Double)
deriving instance Ord (SIMDPACK Double)
deriving instance Fractional (SIMDPACK Double)
deriving instance Floating   (SIMDPACK Double)

-- | RELU and derivative of RELU
relu, relu' :: (Num a, Ord a) => a -> a
relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

-- | derivative of error measurement
cost' :: (Num a, Ord a) => a -> a -> a
cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

tanh, tanh' :: SIMDPACK Float -> SIMDPACK Float
tanh (F x)  = F (Prelude.tanh x)
tanh' (F x) = let a = Prelude.tanh x in F (1 - a*a)
