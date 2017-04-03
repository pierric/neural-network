{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
module Main where
import Test.Hspec
import Test.QuickCheck

main = hspec $ do
  describe "vec + vec" $ do
    it "zero + vec1 = vec1" $ do
      forAll (tensor $ D1 5) $
        \t -> zero :.+ t 
