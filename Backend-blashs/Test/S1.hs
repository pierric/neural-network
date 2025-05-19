{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
module Main where
import Test.Hspec
import Test.QuickCheck hiding (scale)
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import qualified Data.Vector.Storable as V
import Test.Gen
import Test.Utils

main = hspec $ do
  describe "Corr Single" $ do
    it "implements corr2 correct.0" $ do
      forAll (pair (squared_real_matrices 3) (squared_real_matrices 4)) $
        \(m1, m2) -> ioProperty $ do
          r <- test_corr2 2 m1 m2
          return $ good_corr2 2 m1 m2 `eqShowWhenFail` r
    it "implements corr2 correct.1" $ do
      forAll (pair (squared_real_matrices 9) (squared_real_matrices 9)) $
        \(m1, m2) -> ioProperty $ do
          r <- test_corr2 4 m1 m2
          return $ good_corr2 4 m1 m2 `eq` r
    it "implements corr2 correct.2" $ do
      forAll (pair (squared_real_matrices 5) (squared_real_matrices 28)) $
        \(m1, m2) -> ioProperty $ do
          r <- test_corr2 2 m1 m2
          return $ eqShowWhenFail (good_corr2 2 m1 m2) r
    it "implements corr2 correct.3" $ do
      forAll (pair (choose (2,30)) (pair small_matrices small_matrices)) $
        \(p, (m1, m2)) -> ioProperty $ do
          r <- test_corr2 p m1 m2
          return $ good_corr2 p m1 m2 `eqShowWhenFail` r
  describe "Corr Many" $ do
    it "with 2 kernels" $ do
      forAll (pair (sequence $ replicate 2 $ squared_real_matrices 3) (squared_real_matrices 20)) $
        \(m1s, m2) -> ioProperty $ do
          rs <- test_corr2_arr 2 m1s m2
          return $ conjoin $ zipWith (\m r -> good_corr2 2 m m2 `eqShowWhenFail` r) m1s rs
    it "with 5 kernels" $ do
      forAll (pair (sequence $ replicate 4 $ squared_real_matrices 15) (squared_real_matrices 32)) $
        \(m1s, m2) -> ioProperty $ do
          rs <- test_corr2_arr 2 m1s m2
          ss <- return $ map (\m -> good_corr2 2 m m2) m1s
          return $ conjoin $ zipWith eqShowWhenFail rs ss

eqShowWhenFailWithTol :: (Numeric a, Ord a, Show a, Num (Vector a)) => a -> a -> Matrix a -> Matrix a -> Property
eqShowWhenFailWithTol atol rtol m1 m2 =
    let diff = abs (m1 - m2) - scale rtol (abs m2)
        midx = maxIndex diff
        v1 = atIndex m1 midx
        v2 = atIndex m2 midx
        merr = atIndex diff midx
    in whenFail (do
        putStrLn $ "Max difference: " ++ show merr
        putStrLn $ "  " ++ show v1 ++ " vs. " ++ show v2
      ) $ maxElement diff < atol

eqShowWhenFail = eqShowWhenFailWithTol 1e-3 1e-2
