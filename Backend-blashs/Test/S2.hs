module Main where
import Data.NeuralNetwork.Backend.BLASHS.Utils
import qualified Data.Vector.Storable as SV
import Numeric.LinearAlgebra
import Test.Utils

main = t1

t1 = do
  let v1 = SV.fromList [1,2,3,4,
                        5,6,7,8,
                        9,8,7,6,
                        5,4,3,2] :: SV.Vector Float
  let v2 = SV.fromList [0,1,0,
                        1,0,1,
                        0,1,0] :: SV.Vector Float
  m1 <- DenseMatrix 4 4 <$> SV.thaw v1
  m2 <- DenseMatrix 3 3 <$> SV.thaw v2
  let p = 8
  wrk <- newDenseMatrix ((4-3+2*p+1)*(4-3+2*p+1)) (3*3)
  make wrk p m1 4 4 3 3
  putStrLn $ show $ asHM wrk

t2 = do
  let m1 = (3><3) [11.69,  13.02,  -87.05,
                   14.79,   0.71,    2.38,
                  -22.13,  17.06,    9.76] :: Matrix Float
  let m2 = (4><4) [21.21,  -14.38,  -16.15,   17.92,
                   -4.15,  -42.14,   14.31,  -62.53,
                    4.40,  116.08,   14.39,   37.07,
                   -7.53,  -77.93,   12.39,   -2.97] :: Matrix Float
  m3 <- test_corr2 8 m1 m2
  putStrLn $ show m3
