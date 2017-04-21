module Main where

import qualified Data.Vector.Storable as PV
import Control.Monad
import Test.QuickCheck
import Data.List
import Text.PrettyPrint.Free (pretty)
import Data.Tensor (Tensor(..), D1(..), D2(..))
import Data.Tensor.Compile
import Gen
import Comp
import Hmatrix

main = do
  -- let s = 3
  -- t1 <- packTensor (D2 s s) $ hm2v (ident s)
  -- t2 <- packTensor (D1 s) $ PV.fromList [2.0,3.0,4.0]
  -- t3 <- execute' $ I t1 :#> I t2
  -- putStrLn $ show $ t2hv t3

  -- let d1 = D1 2
  --     d2 = D2 2 6
  --     d3 = D1 6
  --     v1 = PV.fromList [-99.856834,-93.86506]
  --     v2 = PV.fromList [121.249306,103.03652,-12.633141,88.78463,-79.31854,-1598.8282,-717.3386,-6.342739,50.755314,17.417084,-104.23368,-345.42133]
  -- t1 <- packTensor d1 v1
  -- t2 <- packTensor d2 v2
  -- t3 <- packTensor d3 $ hv2v (v2hv d1 v1 <# v2hm d2 v2)
  -- t4 <- eval' $ I t1 :<# I t2
  -- putStrLn $ show $ t2hv t3
  -- putStrLn $ show $ t2hv t4

  -- es <- generate (sequence $ replicate 40 $ resize 20 $ (arbitrary :: Gen (Expr D1 Float)))
  -- print $ sort $ map esize es

  -- t1 <- generate $ genTensor (D1 11)
  -- t2 <- generate $ genTensor (D1 11)
  -- t3 <- generate $ genTensor (D2 11 4)
  -- t4 <- generate $ genTensor (D2 4 8)
  -- t5 <- generate $ genTensor (D2 11 8)
  -- t6 <- generate $ genTensor (D1 4)
  -- t7 <- generate $ genTensor (D1 4)
  -- t8 <- generate $ genTensor (D1 4)
  -- t9 <- generate $ genTensor (D1 4)
  -- t10<- generate $ genTensor (D1 4)
  -- let e :: Expr D1 Float
  --     e = (((I t1 :.+ I t2) :<# (S (-26.0) (I t3 :.+ (I t4 :%# I t5)))) :.+ (I t6 :.* I t7) ) :.+ (I t8 :.* (I t9 :.+ I t10))
  -- sequence_ $ replicate 500 $ eval e

  t1 <- generate $ genTensor (D2 4 9) :: IO (Tensor D2 Float)
  t2 <- generate $ genTensor (D2 4 9) :: IO (Tensor D2 Float)
  t3 <- generate $ genTensor (D2 4 9) :: IO (Tensor D2 Float)
  let oe = (DimWrap (D2 4 9),11111111) :@ Bin DM
              ((DimWrap (D2 4 9), 22222222) :@ S 1
                  ((DimWrap (D2 4 9), 333333) :@ I (TensorWrap t1)))
              ((DimWrap (D2 4 9), 444) :@ Bin DM
                  ((DimWrap (D2 4 9), 55) :@ I (TensorWrap t2))
                  ((DimWrap (D2 4 9), 66) :@ I (TensorWrap t3)))
  se <- generate (resize 2 arbitrary)
  (ie,_) <- insert_ce 2 se oe
  let ee = fst $ elimCommonExpr ie
  let df = diff_ce ee ie

  putStrLn $ show $ pretty oe
  putStrLn $ "------------------"
  putStrLn $ show $ pretty se
  putStrLn $ "------------------"
  putStrLn $ show $ pretty ie
  putStrLn $ "------------------"
  putStrLn $ show $ pretty ee
  putStrLn $ "------------------"
  putStrLn $ show $ pretty df
  return ()
