{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
module Main where

import Test.Hspec
import Test.QuickCheck
import qualified Data.Vector.Storable as PV
import Data.Tensor

mkZ :: Dimension d => d -> IO (Tensor d Float)
mkZ = newTensor

mkV :: Dimension d => d -> Gen (PV.Vector Float)
mkV d = PV.fromList <$> vectorOf (size d) arbitrary

eq :: (Dimension d, Element a, Eq a) => Tensor d a -> Tensor d a -> IO Bool
eq t1 t2 = do
  d1 <- PV.unsafeFreeze (_tdat t1)
  d2 <- PV.unsafeFreeze (_tdat t2)
  return $ _tdim t1 == _tdim t2 && d1 == d2

main = hspec $ do
  describe "vec + vec" $ do
    it "zero + vec1 = vec1" $ do
      let d = D1 5
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- packTensor d v
          t2 <- execute' $ I zr :.+ I t1
          eq t1 t2
    it "vec1 + zero = vec1" $ do
      let d = D1 5
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- packTensor d v
          t2 <- execute' $ I t1 :.+ I zr
          eq t1 t2
    it "zero * vec1 = zero" $ do
      let d = D1 3
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- packTensor d v
          t2 <- execute' $ I zr :.* I t1
          eq zr t2
    it "vec1 * zero = zero" $ do
      let d = D1 3
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- I <$> packTensor d v
          t2 <- execute' $ I zr :.* t1
          eq zr t2
