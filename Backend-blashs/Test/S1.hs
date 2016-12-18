{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
module Main where
import Test.Hspec
import Test.QuickCheck
import Numeric.LinearAlgebra
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
      forAll (pair (sequence $ replicate 2 $ squared_real_matrices 3) (squared_real_matrices 7)) $
        \(m1s, m2) -> ioProperty $ do
          rs <- test_corr2_arr 2 m1s m2
          return $ conjoin $ zipWith (\m r -> good_corr2 2 m m2 `eqShowWhenFail` r) m1s rs
    it "with 5 kernels" $ do
      forAll (pair (sequence $ replicate 4 $ squared_real_matrices 15) (squared_real_matrices 88)) $
        \(m1s, m2) -> ioProperty $ do
          rs <- test_corr2_arr 2 m1s m2
          ss <- return $ map (\m -> good_corr2 2 m m2) m1s
          return $ conjoin $ zipWith eq rs ss

eqShowWhenFail m1 m2 =
    whenFail (do let va = flatten m1
                 let vb = flatten m2
                 let err x 0 = x
                     err x y = abs ((x - y) / y)
                 let ev = (V.zipWith err va vb)
                     ei = V.maxIndex ev
                 putStrLn $ "Max error ration: " ++ show (ev V.! ei, va V.! ei, vb V.! ei)
                 putStrLn $ show m1
                 putStrLn $ show m2)
        (m1 `eq` m2)
