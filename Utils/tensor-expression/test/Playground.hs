module Main where

import qualified Data.Vector.Storable as PV
import Control.Monad
import Test.QuickCheck
import Data.List
import Text.PrettyPrint.Free (Pretty(..), encloseSep, lbracket, rbracket, comma, text, renderPretty, displayS)
import Text.Printf (printf)
import Control.Monad.Trans (liftIO)
import Data.Tensor (Tensor(..), D1(..), D2(..), Element, initCG, runCG, newTensor, packTensor)
import Data.Tensor.Compile (ExprHashed, ExprAttr(..), ExprBody(..), ExprOp(..), DimWrap(..), TensorWrap(..), Var(..), attrDim, mkExprHashed)
import Data.Tensor.Execute (Statement(..), Order(..), Transpose(..), toStatements, execute, execute')
import qualified Data.Tensor.Execute as E (Var(..))
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.ByteString.Lazy as BS
import Data.Binary (encode, decode)
import qualified Data.Map as M
import Data.List (sortOn)
import Gen
import Comp
import Hmatrix

main0 = do
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

-- check CSE

--   t1 <- generate $ genTensor (D2 4 9) :: IO (Tensor D2 Float)
--   t2 <- generate $ genTensor (D2 4 9) :: IO (Tensor D2 Float)
--   t3 <- generate $ genTensor (D2 4 9) :: IO (Tensor D2 Float)
--   let oe = (DimWrap (D2 4 9),11111111) :@ Bin DM
--               ((DimWrap (D2 4 9), 22222222) :@ S 1
--                   ((DimWrap (D2 4 9), 333333) :@ I (TensorWrap t1)))
--               ((DimWrap (D2 4 9), 444) :@ Bin DM
--                   ((DimWrap (D2 4 9), 55) :@ I (TensorWrap t2))
--                   ((DimWrap (D2 4 9), 66) :@ I (TensorWrap t3)))
--   se <- generate (resize 2 arbitrary)
--   (ie,_) <- insert_ce 2 se oe
--   let ee = fst $ elimCommonExpr ie
--   let df = diff_ce ee ie

--   putStrLn $ show $ pretty oe
--   putStrLn $ "------------------"
--   putStrLn $ show $ pretty se
--   putStrLn $ "------------------"
--   putStrLn $ show $ pretty ie
--   putStrLn $ "------------------"
--   putStrLn $ show $ pretty ee
--   putStrLn $ "------------------"
--   putStrLn $ show $ pretty df

-- NAN

  let gsz = 3
  t1 <- generate $ resize gsz $ genTensor (D2 24 21) :: IO (Tensor D2 Float)
  t2 <- generate $ resize gsz $ genTensor (D2 24 21) :: IO (Tensor D2 Float)
  t3 <- generate $ resize gsz $ genTensor (D1    7 ) :: IO (Tensor D1 Float)
  t4 <- generate $ resize gsz $ genTensor (D1    29) :: IO (Tensor D1 Float)
  t5 <- generate $ resize gsz $ genTensor (D2 29 24) :: IO (Tensor D2 Float)
  t6 <- generate $ resize gsz $ genTensor (D2 26 39) :: IO (Tensor D2 Float)
  t7 <- generate $ resize gsz $ genTensor (D2 21 39) :: IO (Tensor D2 Float)
  t8 <- generate $ resize gsz $ genTensor (D2 24 23) :: IO (Tensor D2 Float)
  t9 <- generate $ resize gsz $ genTensor (D1    23) :: IO (Tensor D1 Float)
  t10<- generate $ resize gsz $ genTensor (D2 21 26) :: IO (Tensor D2 Float)
  t11<- generate $ resize gsz $ genTensor (D2 21 26) :: IO (Tensor D2 Float)

  let oe = ltt (D1 7) (Var 1) (sca (D2 24 21) (-1000.2) (imm (D2 24 21) t1)) $
           ltt (D1 7) (Var 2) (dm  (D2 24 21) (imm (D2 24 21) t2) (var (D2 24 21) (Var 1))) $
           (mv (D1 7) 
               (ovv (D2 7 26)
                 (imm (D1 7) t3)
                 (vm  (D1 26) 
                    (vm (D1 21)
                       (vm (D1 24)
                          (imm (D1 29) t4)
                          (imm (D2 29 24) t5))
                       (var (D2 24 21) (Var 2)))
                    (mtm (D2 21 26)
                          (imm (D2 26 39) t6)
                          (imm (D2 21 39) t7))))
               (vm  (D1 26) 
                 (vm (D1 21)
                    (mv (D1 24)
                       (imm (D2 24 23) t8)
                       (imm (D1 23)    t9))
                    (var (D2 24 21) (Var 2)))
                 (da (D2 21 26)
                    (imm (D2 21 26) t10)
                    (imm (D2 21 26) t11))))
  TensorWrap ot <- evalExprHashed oe
  d <- PV.unsafeFreeze (_tdat t1)
  print d

evalExprHashed :: Element e => ExprHashed e -> IO (TensorWrap e)
evalExprHashed e = do
  case attrDim e of
    DimWrap d -> do
      t <- newTensor d
      handleE $ runCG initCG $ do
        (st, v) <- toStatements e
        liftIO $ execute $ st ++ [Store v (TensorWrap t)]
      return (TensorWrap t)
  where
    handleE act = act >>= either (ioError . userError . show) return

imm d t   = mkExprHashed (DimWrap d) (I $ TensorWrap t)
var d v   = mkExprHashed (DimWrap d) (V v)
ltt d v e1 e2 = mkExprHashed (DimWrap d) (L v e1 e2)
sca d f e = mkExprHashed (DimWrap d) (S f e)
dm  d x y = mkExprHashed (DimWrap d) (Bin DM  x y)
da  d x y = mkExprHashed (DimWrap d) (Bin DA  x y)
mv  d x y = mkExprHashed (DimWrap d) (Bin MV  x y)
vm  d x y = mkExprHashed (DimWrap d) (Bin VM  x y)
mtm d x y = mkExprHashed (DimWrap d) (Bin MTM x y)
ovv d x y = mkExprHashed (DimWrap d) (Bin OVV x y)

------------------------------------------------------------------------------ 

v000 = E.Var 0
v001 = E.Var 1
v002 = E.Var 2
v003 = E.Var 3
v004 = E.Var 4
v005 = E.Var 5
v006 = E.Var 6
v007 = E.Var 7
v008 = E.Var 8
v009 = E.Var 9
v010 = E.Var 10
v011 = E.Var 11
v012 = E.Var 12
v013 = E.Var 13
v014 = E.Var 14
v015 = E.Var 15
v016 = E.Var 16
v017 = E.Var 17
v018 = E.Var 18
v019 = E.Var 19
v020 = E.Var 20
v021 = E.Var 21
v022 = E.Var 22 
v023 = E.Var 23
v024 = E.Var 24
v025 = E.Var 25 
v026 = E.Var 26
v027 = E.Var 27
v028 = E.Var 28
v029 = E.Var 29
v030 = E.Var 30
v031 = E.Var 31
v032 = E.Var 32
v033 = E.Var 33
v034 = E.Var 34
v035 = E.Var 35
v036 = E.Var 36
v037 = E.Var 37

t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19 :: TensorWrap Float
t1 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t1"  >>= packTensor (D2 26 12) . decode)
t2 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t2"  >>= packTensor (D2 26 12) . decode)
t3 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t3"  >>= packTensor (D1 39   ) . decode)
t4 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t4"  >>= packTensor (D1 12   ) . decode)
t5 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t5"  >>= packTensor (D1 12   ) . decode)
t6 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t6"  >>= packTensor (D1 26   ) . decode)
t7 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t7"  >>= packTensor (D2 26 33) . decode)
t8 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t8"  >>= packTensor (D1 33   ) . decode)
t9 = TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t9"  >>= packTensor (D2 26 12) . decode)
t10= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t10" >>= packTensor (D2 26 12) . decode)
t11= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t11" >>= packTensor (D2  3 12) . decode)
t12= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t12" >>= packTensor (D2 12 12) . decode)
t13= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t13" >>= packTensor (D2 26 12) . decode)
t14= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t14" >>= packTensor (D2 26 12) . decode)
t15= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t15" >>= packTensor (D1 39   ) . decode)
t16= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t16" >>= packTensor (D1 12   ) . decode)
t17= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t17" >>= packTensor (D2 26 29) . decode)
t18= TensorWrap $ unsafePerformIO $ (BS.readFile "D:\\Dev\\NN\\neural-network\\Utils\\tensor-expression\\t18" >>= packTensor (D2 3  29) . decode)
t19= TensorWrap $ unsafePerformIO $ newTensor  (D1 12   )


statements = [ Bind     v000 t1 -- <tensor (D2 26 12): 0x0000000004ccf430 +  312>
             , Bind     v001 t2 -- <tensor (D2 26 12): 0x00000000086ad650 +  312>
             , Alloc    (DimWrap (D2 26 12)) v002
             , DotMul   v000 v001 v002
             , Bind     v003 t3 -- <tensor (D1 39   ): 0x00000000086ac0b0 +   39>
             , Bind     v004 t4 -- <tensor (D1 12   ): 0x00000000086ac2b0 +   12>
             , Alloc    (DimWrap (D2 39 12)) v005
             , BlasGERU RowMajor (D1 39,v003) (D1 12,v004) 1.0 (D2 39 12,v005)
             , Alloc    (DimWrap (D2 39 26)) v006
             , BlasGEMM ColMajor Trans (D2 26 12,v002) NoTrans (D2 39 12,v005) 1.0 0.0 (D2 39 26,v006)
             , Bind     v007 t5 -- <tensor (D1 12   ): 0x00000000086acb60 +   12>
             , Bind     v008 t6 -- <tensor (D1 26   ): 0x00000000086acd00 +   26>
             , Alloc    (DimWrap (D2 12 26)) v009
             , BlasGERU RowMajor (D1 12,v007) (D1 26,v008) 1.0 (D2 12 26,v009)
             , Alloc    (DimWrap (D2 12 39)) v010
             , BlasGEMM ColMajor Trans (D2 39 26,v006) NoTrans (D2 12 26,v009) 1.0 0.0 (D2 12 39,v010)
             , Bind     v011 t7 -- <tensor (D2 26 33): 0x0000000004cbe010 +  858>
             , Bind     v012 t8 -- <tensor (D1 33   ): 0x0000000008698110 +   33>
             , Alloc    (DimWrap (D1 26)) v013
             , BlasGEMV RowMajor NoTrans (D2 26 33,v011) (D1 33,v012) 1.0 0.0 (D1 26,v013)
             , Bind     v014 t9 -- <tensor (D2 26 12): 0x0000000004ccf430 +  312>
             , Bind     v015 t10 -- <tensor (D2 26 12): 0x00000000086ad650 +  312>
             , Alloc    (DimWrap (D2 26 12)) v016
             , DotMul   v014 v015 v016
             , Alloc    (DimWrap (D1 12)) v017
             , BlasGEMV RowMajor Trans (D2 26 12,v016) (D1 26,v013) 1.0 0.0 (D1 12,v017)
             , Bind     v018 t11 -- <tensor (D2 3 12 ): 0x0000000008698960 +   36>
             , Bind     v019 t12 -- <tensor (D2 12 12): 0x0000000004c68010 +  144>
             , Alloc    (DimWrap (D2 12 3)) v020
             , BlasGEMM ColMajor Trans (D2 3 12,v018) NoTrans (D2 12 12,v019) 1.0 0.0 (D2 12 3,v020)
             , Alloc    (DimWrap (D1 3)) v021
             , BlasGEMV RowMajor Trans (D2 12 3,v020) (D1 12,v017) 1.0 0.0 (D1 3,v021)
             , Bind     v022 t13 -- <tensor (D2 26 12): 0x0000000004ccf430 +  312>
             , Bind     v023 t14 -- <tensor (D2 26 12): 0x00000000086ad650 +  312>
             , Alloc    (DimWrap (D2 26 12)) v024
             , DotMul   v022 v023 v024
             , Bind     v025 t15 -- <tensor (D1 39   ): 0x0000000004c68bf0 +   39>
             , Bind     v026 t16 -- <tensor (D1 12   ): 0x0000000004c68df0 +   12>
             , Alloc    (DimWrap (D2 39 12)) v027
             , BlasGERU RowMajor (D1 39,v025) (D1 12,v026) 1.0 (D2 39 12,v027)
             , Alloc    (DimWrap (D2 39 26)) v028
             , BlasGEMM ColMajor Trans (D2 26 12,v024) NoTrans (D2 39 12,v027) 1.0 0.0 (D2 39 26,v028)
             , Bind     v029 t17 -- <tensor (D2 26 29): 0x0000000004ce0010 +  754>
             , Bind     v030 t18 -- <tensor (D2 3 29 ): 0x0000000004c53010 +   87>
             , Alloc    (DimWrap (D2 3 26)) v031
             , BlasGEMM ColMajor Trans (D2 26 29,v029) NoTrans (D2 3 29,v030) 1.0 0.0 (D2 3 26,v031)
             , Alloc    (DimWrap (D2 3 39)) v032
             , BlasGEMM ColMajor Trans (D2 39 26,v028) NoTrans (D2 3 26,v031) 1.0 0.0 (D2 3 39,v032)
             , Alloc    (DimWrap (D1 39)) v033
             , BlasGEMV RowMajor Trans (D2 3 39,v032) (D1 3,v021) 1.0 0.0 (D1 39,v033)
             , Alloc    (DimWrap (D1 12)) v034
             , BlasGEMV RowMajor NoTrans (D2 12 39,v010) (D1 39,v033) 1.0 0.0 (D1 12,v034)
             , Store    v034 t19
             ]

prTensor (TensorWrap t) = do
  d <- PV.unsafeFreeze (_tdat t)
  -- putStrLn $ show d
  let pp = encloseSep lbracket rbracket comma $ map (text . printf "%.1e") (PV.toList d)
  putStrLn $ displayS (renderPretty 0.4 100000 pp) ""

v10, v33 :: [Float]
-- v10 = [-1.95414e12,3.0685677e12,7.9979694e12,2.8677754e12,-1.542466e12,-2.8200122e12,2.513288e12,3.3877456e11,-2.1112582e12,9.248937e11,1.8271609e12,9.826564e12,1.1609147e11,6.005395e11,-3.1212904e12,4.5393446e13,-1.0235835e13,5.736044e12,2.1288967e12,6.69206e12,9.666889e12,3.0047926e12,-5.807124e12,8.915028e12,7.1087974e11,-1.7367733e12,-1.7695102e13,-2.1783212e12,3.988644e11,1.0447056e12,-2.9602108e12,-5.545089e11,-4.952854e12,4.1376865e11,8.7570547e11,-4.7837206e12,6.8283014e11,-3.6895303e12,5.901012e12,1.5557347e11,-2.4429553e11,-6.3673644e11,-2.2831006e11,1.2279917e11,2.2450754e11,-2.0008852e11,-2.6970601e10,1.6808205e11,-7.363285e10,-1.4546431e11,-7.823149e11,-9.242299e9,-4.7810294e10,2.4849287e11,-3.6138745e12,8.148978e11,-4.566594e11,-1.6948627e11,-5.3277006e11,-7.696027e11,-2.3921825e11,4.6231817e11,-7.0974544e11,-5.659472e10,1.382684e11,1.408747e12,1.7342105e11,-3.175449e10,-8.317132e10,2.3566895e11,4.4145697e10,3.943078e11,-3.2941054e10,-6.971687e10,3.8084277e11,-5.4361645e10,2.9373173e11,-4.6979288e11,-2.2585221e11,3.5465357e11,9.243757e11,3.3144685e11,-1.7827253e11,-3.259265e11,2.9047652e11,3.9154303e10,-2.4401142e11,1.0689577e11,2.1117644e11,1.1357178e12,1.3417422e10,6.940813e10,-3.6074704e11,5.2464053e12,-1.1830198e12,6.629509e11,2.4604996e11,7.734435e11,1.1172628e12,3.4728267e11,-6.7116584e11,1.0303661e12,8.216082e10,-2.007298e11,-2.0451345e12,-2.5176225e11,4.6099268e10,1.2074311e11,-3.4213007e11,-6.4088048e10,-5.7243245e11,4.7821816e10,1.0121076e11,-5.528847e11,7.891899e10,-4.264221e11,6.8201716e11,1.366969e11,-2.1465383e11,-5.594776e11,-2.0060796e11,1.0789929e11,1.9726677e11,-1.7581069e11,-2.3698106e10,1.4768769e11,-6.4698577e10,-1.2781436e11,-6.873924e11,-8.120882e9,-4.2009207e10,2.1834185e11,-3.1753836e12,7.1602176e11,-4.012505e11,-1.4892155e11,-4.6812607e11,-6.762226e11,-2.101926e11,4.0622257e11,-6.2362786e11,-4.972778e10,1.2149156e11,1.2378162e12,1.5237888e11,-2.790154e10,-7.3079685e10,2.0707398e11,3.8789267e10,3.464643e11,-2.894413e10,-6.1257757e10,3.34633e11,-4.7765647e10,2.5809165e11,-4.1279033e11,6.7202764e12,-1.0552786e13,-2.7504968e13,-9.862264e12,5.3045326e12,9.698002e12,-8.643181e12,-1.165044e12,7.2606045e12,-3.1807035e12,-6.2835954e12,-3.3793495e13,-3.992378e11,-2.0652514e12,1.0734098e13,-1.5610782e14,3.5200977e13,-1.9726216e13,-7.3212614e12,-2.3013947e13,-3.3244366e13,-1.0333462e13,1.9970665e13,-3.0658726e13,-2.4447106e12,5.972753e12,6.0853357e13,7.4912346e12,-1.3716917e12,-3.5927339e12,1.0180144e13,1.9069525e12,1.7032834e13,-1.4229475e12,-3.0115451e12,1.645119e13,-2.3482485e12,1.2688269e13,-2.0293546e13,3.2767997e11,-5.1455282e11,-1.3411395e12,-4.8088295e11,2.5864844e11,4.728738e11,-4.214406e11,-5.6807416e10,3.540264e11,-1.5509076e11,-3.0638745e11,-1.6477675e12,-1.94668e10,-1.00701454e11,5.2339353e11,-7.611801e12,1.7163961e12,-9.618487e11,-3.569841e11,-1.122158e12,-1.6209921e12,-5.038587e11,9.737675e11,-1.4949163e12,-1.1920386e11,2.912308e11,2.967203e12,3.6527188e11,-6.6883568e10,-1.7518137e11,4.963829e11,9.298283e10,8.305191e11,-6.938279e10,-1.4684268e11,8.021583e11,-1.1450039e11,6.1867865e11,-9.8951155e11,4.4620743e11,-7.0067526e11,-1.8262525e12,-6.548267e11,3.5220586e11,6.439202e11,-5.7388315e11,-7.735562e10,4.8208373e11,-2.1118978e11,-4.1721302e11,-2.2437935e12,-2.650827e10,-1.3712688e11,7.127139e11,-1.0365116e13,2.3372463e12,-1.3097658e12,-4.8611135e11,-1.5280615e12,-2.2073332e12,-6.8611283e11,1.3259962e12,-2.0356534e12,-1.623219e11,3.9657393e11,4.0404913e12,4.9739688e11,-9.10765e10,-2.3854748e11,6.7593306e11,1.2661629e11,1.1309322e12,-9.447972e10,-1.9995815e11,1.0923124e12,-1.5591714e11,8.424654e11,-1.3474347e12,1.1638738e11,-1.8276201e11,-4.763543e11,-1.7080297e11,9.1868365e10,1.6795823e11,-1.4968994e11,-2.0177207e10,1.2574526e11,-5.5086105e10,-1.0882458e11,-5.8526414e11,-6.9143357e9,-3.576776e10,1.8590215e11,-2.7036056e12,6.0964025e11,-3.4163527e11,-1.2679577e11,-3.9857504e11,-5.757541e11,-1.7896363e11,3.4586873e11,-5.3097362e11,-4.233955e10,1.0344117e11,1.0539094e12,1.2973948e11,-2.375612e10,-6.222199e10,1.7630829e11,3.3026204e10,2.9498884e11,-2.4643805e10,-5.215648e10,2.8491543e11,-4.066895e10,2.1974617e11,-3.5146075e11,3.2488184e11,-5.1015893e11,-1.3296873e12,-4.7677656e11,2.564398e11,4.6883573e11,-4.1784184e11,-5.6322322e10,3.5100325e11,-1.537664e11,-3.0377106e11,-1.6336967e12,-1.9300565e10,-9.984156e10,5.1892408e11,-7.5468023e12,1.7017394e12,-9.536351e11,-3.5393562e11,-1.112576e12,-1.6071503e12,-4.9955606e11,9.654522e11,-1.482151e12,-1.1818594e11,2.8874388e11,2.941866e12,3.621528e11,-6.631244e10,-1.7368544e11,4.9214416e11,9.2188836e10,8.234272e11,-6.879032e10,-1.4558875e11,7.953082e11,-1.1352263e11,6.133958e11,-9.8106186e11,2.8834316e11,-4.527826e11,-1.1801408e12,-4.2315478e11,2.2759868e11,4.1610687e11,-3.7084824e11,-4.998789e10,3.115268e11,-1.36472674e11,-2.6960675e11,-1.4499591e12,-1.7129875e10,-8.861261e10,4.6056204e11,-6.698032e12,1.5103487e12,-8.4638237e11,-3.1412935e11,-9.874472e11,-1.4263982e12,-4.433722e11,8.5687035e11,-1.3154571e12,-1.0489386e11,2.5626963e11,2.6110012e12,3.2142236e11,-5.8854445e10,-1.541515e11,4.3679387e11,8.1820566e10,7.308182e11,-6.1053645e10,-1.2921475e11,7.058619e11,-1.0075504e11,5.4440867e11,-8.7072394e11,7.635625e11,-1.1990151e12,-3.1251348e12,-1.1205576e12,6.027047e11,1.101894e12,-9.820443e11,-1.3237311e11,8.249551e11,-3.613938e11,-7.139464e11,-3.8396413e12,-4.5361684e10,-2.3465535e11,1.2196158e12,-1.7737078e13,3.9995606e12,-2.2413076e12,-8.318474e11,-2.6148625e12,-3.7772494e12,-1.1740959e12,2.2690818e12,-3.483467e12,-2.7776978e11,6.7862836e11,6.914202e12,8.5115935e11,-1.5585267e11,-4.0820897e11,1.1566751e12,2.1666934e11,1.9352828e12,-1.6167638e11,-3.4217397e11,1.8691954e12,-2.6680972e11,1.4416503e12,-2.3057675e12,7.709481e11,-1.2106122e12,-3.1553613e12,-1.1313954e12,6.0853413e11,1.112552e12,-9.915431e11,-1.3365342e11,8.3293425e11,-3.6488928e11,-7.2085176e11,-3.8767784e12,-4.580044e10,-2.36925e11,1.2314123e12,-1.7908636e13,4.0382455e12,-2.2629864e12,-8.398928e11,-2.6401531e12,-3.8137834e12,-1.185452e12,2.291028e12,-3.5171583e12,-2.8045646e11,6.8519225e11,6.9810777e12,8.593919e11,-1.573601e11,-4.121572e11,1.1678629e12,2.18765e11,1.9540007e12,-1.6324012e11,-3.4548358e11,1.8872744e12,-2.693904e11,1.455594e12,-2.3280686e12]
-- v33 = [-3.444717e29,2.3803084e30,1.4326192e29,6.550688e29,-5.37489e29,-5.7480327e29,1.6298795e29,1.2081965e30,1.9175957e29,-1.8152784e29,-4.108357e30,2.2038262e29,4.2626238e30,-2.0088442e30,7.541725e29,-7.657183e29,1.525134e30,3.73805e29,4.8185743e29,1.0960459e30,-1.1273613e29,9.1661785e28,-4.218193e29,-1.8275197e31,-8.114909e29,-9.306103e29,8.2677205e29,9.486578e28,2.0654587e28,3.7100184e28,4.077699e29,-3.7027233e29,5.7781312e29,1.7877236e29,-8.2484744e29,-2.4107e29,3.3006276e31,-1.2798408e30,-3.1273932e29]
v10 = [-2.0e12, 3.1e12, 8.0e12, 2.9e12, -1.5e12, -2.8e12, 2.5e12, 3.4e11, -2.1e12, 9.2e11, 1.8e12, 9.8e12, 1.2e11, 6.0e11, -3.1e12, 4.5e13, -1.0e13, 5.7e12, 2.1e12, 6.7e12, 9.7e12, 3.0e12, -5.8e12, 8.9e12, 7.1e11, -1.7e12, -1.8e13, -2.2e12, 4.0e11, 1.0e12, -3.0e12, -5.5e11, -5.0e12, 4.1e11, 8.8e11, -4.8e12, 6.8e11, -3.7e12, 5.9e12, 1.6e11, -2.4e11, -6.4e11, -2.3e11, 1.2e11, 2.2e11, -2.0e11, -2.7e10, 1.7e11, -7.4e10, -1.5e11, -7.8e11, -9.2e9, -4.8e10, 2.5e11, -3.6e12, 8.1e11, -4.6e11, -1.7e11, -5.3e11, -7.7e11, -2.4e11, 4.6e11, -7.1e11, -5.7e10, 1.4e11, 1.4e12, 1.7e11, -3.2e10, -8.3e10, 2.4e11, 4.4e10, 3.9e11, -3.3e10, -7.0e10, 3.8e11, -5.4e10, 2.9e11, -4.7e11, -2.3e11, 3.5e11, 9.2e11, 3.3e11, -1.8e11, -3.3e11, 2.9e11, 3.9e10, -2.4e11, 1.1e11, 2.1e11, 1.1e12, 1.3e10, 6.9e10, -3.6e11, 5.2e12, -1.2e12, 6.6e11, 2.5e11, 7.7e11, 1.1e12, 3.5e11, -6.7e11, 1.0e12, 8.2e10, -2.0e11, -2.0e12, -2.5e11, 4.6e10, 1.2e11, -3.4e11, -6.4e10, -5.7e11, 4.8e10, 1.0e11, -5.5e11, 7.9e10, -4.3e11, 6.8e11, 1.4e11, -2.1e11, -5.6e11, -2.0e11, 1.1e11, 2.0e11, -1.8e11, -2.4e10, 1.5e11, -6.5e10, -1.3e11, -6.9e11, -8.1e9, -4.2e10, 2.2e11, -3.2e12, 7.2e11, -4.0e11, -1.5e11, -4.7e11, -6.8e11, -2.1e11, 4.1e11, -6.2e11, -5.0e10, 1.2e11, 1.2e12, 1.5e11, -2.8e10, -7.3e10, 2.1e11, 3.9e10, 3.5e11, -2.9e10, -6.1e10, 3.3e11, -4.8e10, 2.6e11, -4.1e11, 6.7e12, -1.1e13, -2.8e13, -9.9e12, 5.3e12, 9.7e12, -8.6e12, -1.2e12, 7.3e12, -3.2e12, -6.3e12, -3.4e13, -4.0e11, -2.1e12, 1.1e13, -1.6e14, 3.5e13, -2.0e13, -7.3e12, -2.3e13, -3.3e13, -1.0e13, 2.0e13, -3.1e13, -2.4e12, 6.0e12, 6.1e13, 7.5e12, -1.4e12, -3.6e12, 1.0e13, 1.9e12, 1.7e13, -1.4e12, -3.0e12, 1.6e13, -2.3e12, 1.3e13, -2.0e13, 3.3e11, -5.1e11, -1.3e12, -4.8e11, 2.6e11, 4.7e11, -4.2e11, -5.7e10, 3.5e11, -1.6e11, -3.1e11, -1.6e12, -1.9e10, -1.0e11, 5.2e11, -7.6e12, 1.7e12, -9.6e11, -3.6e11, -1.1e12, -1.6e12, -5.0e11, 9.7e11, -1.5e12, -1.2e11, 2.9e11, 3.0e12, 3.7e11, -6.7e10, -1.8e11, 5.0e11, 9.3e10, 8.3e11, -6.9e10, -1.5e11, 8.0e11, -1.1e11, 6.2e11, -9.9e11, 4.5e11, -7.0e11, -1.8e12, -6.5e11, 3.5e11, 6.4e11, -5.7e11, -7.7e10, 4.8e11, -2.1e11, -4.2e11, -2.2e12, -2.7e10, -1.4e11, 7.1e11, -1.0e13, 2.3e12, -1.3e12, -4.9e11, -1.5e12, -2.2e12, -6.9e11, 1.3e12, -2.0e12, -1.6e11, 4.0e11, 4.0e12, 5.0e11, -9.1e10, -2.4e11, 6.8e11, 1.3e11, 1.1e12, -9.4e10, -2.0e11, 1.1e12, -1.6e11, 8.4e11, -1.3e12, 1.2e11, -1.8e11, -4.8e11, -1.7e11, 9.2e10, 1.7e11, -1.5e11, -2.0e10, 1.3e11, -5.5e10, -1.1e11, -5.9e11, -6.9e9, -3.6e10, 1.9e11, -2.7e12, 6.1e11, -3.4e11, -1.3e11, -4.0e11, -5.8e11, -1.8e11, 3.5e11, -5.3e11, -4.2e10, 1.0e11, 1.1e12, 1.3e11, -2.4e10, -6.2e10, 1.8e11, 3.3e10, 2.9e11, -2.5e10, -5.2e10, 2.8e11, -4.1e10, 2.2e11, -3.5e11, 3.2e11, -5.1e11, -1.3e12, -4.8e11, 2.6e11, 4.7e11, -4.2e11, -5.6e10, 3.5e11, -1.5e11, -3.0e11, -1.6e12, -1.9e10, -1.0e11, 5.2e11, -7.5e12, 1.7e12, -9.5e11, -3.5e11, -1.1e12, -1.6e12, -5.0e11, 9.7e11, -1.5e12, -1.2e11, 2.9e11, 2.9e12, 3.6e11, -6.6e10, -1.7e11, 4.9e11, 9.2e10, 8.2e11, -6.9e10, -1.5e11, 8.0e11, -1.1e11, 6.1e11, -9.8e11, 2.9e11, -4.5e11, -1.2e12, -4.2e11, 2.3e11, 4.2e11, -3.7e11, -5.0e10, 3.1e11, -1.4e11, -2.7e11, -1.4e12, -1.7e10, -8.9e10, 4.6e11, -6.7e12, 1.5e12, -8.5e11, -3.1e11, -9.9e11, -1.4e12, -4.4e11, 8.6e11, -1.3e12, -1.0e11, 2.6e11, 2.6e12, 3.2e11, -5.9e10, -1.5e11, 4.4e11, 8.2e10, 7.3e11, -6.1e10, -1.3e11, 7.1e11, -1.0e11, 5.4e11, -8.7e11, 7.6e11, -1.2e12, -3.1e12, -1.1e12, 6.0e11, 1.1e12, -9.8e11, -1.3e11, 8.2e11, -3.6e11, -7.1e11, -3.8e12, -4.5e10, -2.3e11, 1.2e12, -1.8e13, 4.0e12, -2.2e12, -8.3e11, -2.6e12, -3.8e12, -1.2e12, 2.3e12, -3.5e12, -2.8e11, 6.8e11, 6.9e12, 8.5e11, -1.6e11, -4.1e11, 1.2e12, 2.2e11, 1.9e12, -1.6e11, -3.4e11, 1.9e12, -2.7e11, 1.4e12, -2.3e12, 7.7e11, -1.2e12, -3.2e12, -1.1e12, 6.1e11, 1.1e12, -9.9e11, -1.3e11, 8.3e11, -3.6e11, -7.2e11, -3.9e12, -4.6e10, -2.4e11, 1.2e12, -1.8e13, 4.0e12, -2.3e12, -8.4e11, -2.6e12, -3.8e12, -1.2e12, 2.3e12, -3.5e12, -2.8e11, 6.9e11, 7.0e12, 8.6e11, -1.6e11, -4.1e11, 1.2e12, 2.2e11, 2.0e12, -1.6e11, -3.5e11, 1.9e12, -2.7e11, 1.5e12, -2.3e12]
v33 = [-3.4e29, 2.4e30, 1.4e29, 6.6e29, -5.4e29, -5.7e29, 1.6e29, 1.2e30, 1.9e29, -1.8e29, -4.1e30, 2.2e29, 4.3e30, -2.0e30, 7.5e29, -7.7e29, 1.5e30, 3.7e29, 4.8e29, 1.1e30, -1.1e29, 9.2e28, -4.2e29, -1.8e31, -8.1e29, -9.3e29, 8.3e29, 9.5e28, 2.1e28, 3.7e28, 4.1e29, -3.7e29, 5.8e29, 1.8e29, -8.2e29, -2.4e29, 3.3e31, -1.3e30, -3.1e29]


tt10 = TensorWrap $ unsafePerformIO $ packTensor (D2 12 39) $ PV.fromList v10
tt33 = TensorWrap $ unsafePerformIO $ packTensor (D1 39) $ PV.fromList v33

tt34 :: TensorWrap Float
tt34 = TensorWrap $ unsafePerformIO $ newTensor  (D1 12)
tsta = [ Bind v034 tt34
       , BlasGEMV RowMajor NoTrans (D2 12 39,v010) (D1 39,v033) 1.0 0.0 (D1 12,v034)]

main = do
  exs <- execute' tsta (M.fromList [(10, tt10),(33, tt33)])
  prTensor tt10
  prTensor tt33
  prTensor tt34

main1 = do
  exs <- execute' statements M.empty  
  putStrLn "--------"
  forM_ (sortOn fst $ M.toList exs) $ \(i,TensorWrap t) -> do
    putStrLn $ "Var " ++ show i
    prTensor (TensorWrap t)
    let tfn = "tt" ++ show i
    putStrLn $ "save as " ++ tfn
    d <- PV.unsafeFreeze (_tdat t)
    BS.writeFile tfn $ encode d
