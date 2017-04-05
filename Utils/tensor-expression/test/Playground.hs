module Main where

import qualified Data.Vector.Storable as PV
import Control.Monad
import Data.Tensor
import Hmatrix

main = do
  -- let s = 3
  -- t1 <- packTensor (D2 s s) $ hm2v (ident s)
  -- t2 <- packTensor (D1 s) $ PV.fromList [2.0,3.0,4.0]
  -- t3 <- execute' $ I t1 :#> I t2
  -- putStrLn $ show $ t2hv t3

  let d1 = D1 2
      d2 = D2 2 6
      d3 = D1 6
      v1 = PV.fromList [-99.856834,-93.86506]
      v2 = PV.fromList [121.249306,103.03652,-12.633141,88.78463,-79.31854,-1598.8282,-717.3386,-6.342739,50.755314,17.417084,-104.23368,-345.42133]
  t1 <- packTensor d1 v1
  t2 <- packTensor d2 v2
  t3 <- packTensor d3 $ hv2v (v2hv d1 v1 <# v2hm d2 v2)
  t4 <- eval' $ I t1 :<# I t2
  putStrLn $ show $ t2hv t3
  putStrLn $ show $ t2hv t4
