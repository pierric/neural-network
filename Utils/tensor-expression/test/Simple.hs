{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
module Main where

import Test.Hspec
import Test.QuickCheck
import qualified Data.Vector.Storable as PV
import Control.Monad
import Data.Tensor
import Hmatrix

mkZ :: Dimension d => d -> IO (Tensor d Float)
mkZ = newTensor

mkV :: Dimension d => d -> Gen (PV.Vector Float)
mkV d = let sf = fromIntegral <$> (arbitrary :: Gen Int)
        in PV.fromList <$> vectorOf (size d) sf

eq :: (Dimension d, Element a, Ord a) => Tensor d a -> Tensor d a -> IO Bool
eq t1 t2 = do
  d1 <- PV.unsafeFreeze (_tdat t1)
  d2 <- PV.unsafeFreeze (_tdat t2)
  let df = PV.map abs (PV.zipWith (-) d1 d2)
  return $ _tdim t1 == _tdim t2 && PV.all (<0.01) df

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
    it "vec1 + vec2 = vec3" $ do
      let d = D1 7
      forAll (liftM2 (,) (mkV d) (mkV d)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d v1
          t2 <- packTensor d v2
          t3 <- packTensor d $ PV.zipWith (+) v1 v2
          t4 <- execute' $ I t1 :.+ I t2
          eq t3 t4
  describe "vec * vec" $ do
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
    it "vec1 * vec2 = vec3" $ do
      let d = D1 7
      forAll (liftM2 (,) (mkV d) (mkV d)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d v1
          t2 <- packTensor d v2
          t3 <- packTensor d $ PV.zipWith (*) v1 v2
          t4 <- execute' $ I t1 :.* I t2
          eq t3 t4
  describe "vec * matrix" $ do
    it "vec1 * ident = vec1" $ do
      let s = 3
      forAll (mkV (D1 s)) $ do
        \v1 -> ioProperty $ do
          t1 <- packTensor (D1 s) v1
          t2 <- packTensor (D2 s s) $ hm2v (ident s)
          t3 <- execute' $ I t1 :<# I t2
          eq t3 t1
    it "vec1 * matrix = vec2" $ do
      let d1 = D1 2
          d2 = D2 2 6
          d3 = D1 6
      forAll (liftM2 (,) (mkV d1) (mkV d2)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d1 v1
          t2 <- packTensor d2 v2
          t3 <- packTensor d3 $ hv2v (v2hv d1 v1 <# v2hm d2 v2)
          t4 <- execute' $ I t1 :<# I t2
          eq t3 t4
  describe "matrix * vec" $ do
    it "ident * vec1 = vec1" $ do
      let s = 3
      forAll (mkV (D1 s)) $ do
        \v1 -> ioProperty $ do
          t1 <- packTensor (D2 s s) $ hm2v (ident s)
          t2 <- packTensor (D1 s) v1
          t3 <- execute' $ I t1 :#> I t2
          eq t2 t3
    it "matrix * vec1 = vec2" $ do
      let d1 = D2 6 2
          d2 = D1 2
          d3 = D1 6
      forAll (liftM2 (,) (mkV d1) (mkV d2)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d1 v1
          t2 <- packTensor d2 v2
          t3 <- packTensor d3 $ hv2v (v2hm d1 v1 #> v2hv d2 v2)
          t4 <- execute' $ I t1 :#> I t2
          eq t3 t4
