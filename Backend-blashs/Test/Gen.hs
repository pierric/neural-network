{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
module Test.Gen where
import Test.Utils
import Test.QuickCheck
import Control.Monad
import qualified Numeric.LinearAlgebra as L
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector.Storable as V

squared_real_matrices :: Int -> Gen (L.Matrix Float)
squared_real_matrices k = do
  vs <- sequence (replicate (k*k) arbitrary) :: Gen [Float]
  return $ L.reshape k $ V.fromList vs

small_matrices :: Gen (L.Matrix Float)
small_matrices = do
  n <- choose (2,10)
  squared_real_matrices n

pair :: Gen a -> Gen b -> Gen (a,b)
pair = liftM2 (,)
