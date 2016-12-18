module Main where
import Data.NeuralNetwork.Backend.BLASHS.Utils
import qualified Data.Vector.Storable as SV
import Numeric.LinearAlgebra
import Test.QuickCheck (generate)
import Test.Utils
import Test.Gen

main = t3

t1 = do
  ms <- test_corr2_arr 0 ks mt
  mapM_ (putStrLn . show) ms
  putStrLn "Good ones are:"
  let m4s = map (\k -> good_corr2 0 k mt) ks
  mapM_ (putStrLn . show) m4s

t2 = do
  let m1 = (3><3) [0, 1, 0,
                   1, 0, 1,
                   0 ,1, 0] :: Matrix Float
  let ma = (3><3) [1, 0, 1,
                   0, 1, 0,
                   1 ,0, 1] :: Matrix Float
  let mb = (3><3) [0, 0, 0,
                   0, 0, 0,
                   0 ,0, 1] :: Matrix Float
  let mc = (3><3) [0, 0, 0,
                   0, 1, 0,
                   0 ,0, 0] :: Matrix Float
  let md = (3><3) [1, 0, 0,
                   0, 0, 0,
                   0 ,0, 0] :: Matrix Float
  let m3 = (4><4) [4,3,2,1,
                   1,2,3,4,
                   2,4,6,8,
                   1,3,5,7] :: Matrix Float
  m3s <- test_corr2_arr 0 [m1, ma, mb, mc, md] m3
  mapM_ (putStrLn . show) m3s

t3 = do
   ks <- generate $ sequence $ replicate 4 $ squared_real_matrices 45
   mt <- generate $ squared_real_matrices 488
   test_corr2_arr 2 ks mt

ks = [(3><3)  [  -2.2182608,    -2.65496,   -4.452847
              , -0.50510013,   1.1137468,   0.5881784
              ,   -2.162154, -0.26882064, -0.21511263 ]
     ,(3><3)  [ -168.82103, -235.06078, 4.6464214
              ,  3.0136368,  0.9492494, 605.85254
              ,  -482.5887,  14.835919, -5.846858 ]
     ,(3><3)  [ -1.3539604,    8.014306, -4.851793
              , 0.84038895, -0.77802753, 1.4430395
              ,  0.5361183,  -1.2853559,   1.04196 ]
     ,(3><3)  [ -2.6642451, -2.149097, -4.7066493
              ,   1.040131,   1.52895,  -5.823082
              ,  -5.870729,  2.996159,  1.2088959 ]
     ,(3><3)  [  -2.5846646,   2.2002325, -2.7979004
              ,   0.5225487,   0.9918108, 0.85764366
              , -0.56938225, -0.14684846,  0.8047059 ]]
mt = (7><7)   [  3.4427254,    0.7416965,   1.2416595,  0.8233356,  -1.565333, 0.89987564,  0.8893491
              , -1.3306698,   -3.9727004,  -1.4404364, 8.25734e-2,  1.0835173, -6.5956297, 0.95600057
              ,  -5.312902, -9.969063e-2,  0.94423944, 0.25942376,  -1.177719, -219.11214,  0.7662355
              , -3.0743675,   0.16589901,   -1.338092,  1.2396933, -2.8903048, -0.2627482, 0.64401335
              ,  6.0397525,   -0.6807962, -0.36770278,   1.416498,  1.8280395, -0.5226003, 0.98356235
              , -1.1980977,   -0.5414966, -0.18184663, 0.20523632,  1.2170252,  1.2050351,  1.1030668
              ,   0.673801,   0.95085037,  -2.3031695,  0.4833055,  1.8391184,   1.117064, -1.7435362 ]
