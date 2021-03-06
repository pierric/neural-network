{-# LANGUAGE FlexibleContexts, FlexibleInstances #-}
module Test.Utils where
import Numeric.LinearAlgebra
import Control.Exception
import Control.Monad
import qualified Data.NeuralNetwork.Backend.BLASHS.Utils as U
import qualified Numeric.LinearAlgebra as L
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector as BV
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import System.IO.Unsafe

asHM (U.DenseMatrix r c v) = L.reshape c $ unsafePerformIO $ V.freeze v
asDM m = let (r,c) = size m in U.DenseMatrix r c (unsafePerformIO $ V.thaw $ L.flatten m)

good_corr2 :: Int -> L.Matrix Float -> L.Matrix Float -> L.Matrix Float

good_corr2 p k m | w > s     = good_corr2 p m k
                 | otherwise = corr2 k padded
  where
    (w,h) = L.size k
    (s,t) = L.size m
    padded = fromBlocks [[z,0,0]
                        ,[0,m,0]
                        ,[0,0,z]]
    z = konst 0 (p, p)

test_corr2 :: Int -> L.Matrix Float -> L.Matrix Float -> IO (L.Matrix Float)
test_corr2 p k m | w > s     = test_corr2 p m k
                 | otherwise = do x@(U.DenseMatrixArray _ _ _ vx) <- U.newDenseMatrixArray 1 r c
                                  k' <- U.DenseMatrix w h <$> V.thaw (flatten k)
                                  m' <- U.DenseMatrix s t <$> V.thaw (flatten m)
                                  U.corr2 p (BV.singleton k') m' (x U.<<=)
                                  reshape c <$> V.freeze vx
  where
    (w,h) = L.size k
    (s,t) = L.size m
    (r,c) = (s-w+2*p+1, t-h+2*p+1)

test_corr2_arr :: Int -> [L.Matrix Float] -> L.Matrix Float -> IO [L.Matrix Float]
test_corr2_arr p ks m = do x@(U.DenseMatrixArray _ _ _ vx) <- U.newDenseMatrixArray n r c
                           print ("test", n, r, c, MV.length vx)
                           ks' <- mapM (\k -> U.DenseMatrix w h <$> V.thaw (flatten k)) ks
                           m'  <- U.DenseMatrix s t <$> V.thaw (flatten m)
                           U.corr2 p (BV.fromList ks') m' (x U.<<=)
                           let vm = U.denseMatrixArrayToVector x
                           vhm <- BV.mapM (\(U.DenseMatrix _ _ vx) -> reshape c <$> V.freeze vx) vm
                           return $ BV.toList vhm

  where
    n     = length ks
    (w,h) = L.size (head ks)
    (s,t) = L.size m
    (r,c) = (s-w+2*p+1, t-h+2*p+1)

eq :: L.Matrix Float -> L.Matrix Float -> Bool
eq a b = V.all id $ ratio a b

ratio a b =
  let va = flatten a
      vb = flatten b
      ae :: V.Vector Float
      ae = V.zipWith (\a b -> abs (a - b)) va vb
      aa = V.sum ae / fromIntegral (V.length ae)
      err x 0 = x < 0.1
      err x y = let e = x-y
                in (abs (e / y) < 0.02)
  in V.zipWith err va vb
