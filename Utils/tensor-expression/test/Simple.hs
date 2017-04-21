{-# LANGUAGE FlexibleInstances, FlexibleContexts, ScopedTypeVariables #-}
module Main where

import Test.Hspec
import Test.QuickCheck hiding (scale)
import qualified Data.Vector.Storable as PV
import Data.Typeable (cast)
import Control.Monad
import Control.Monad.Trans (liftIO)
import Data.Tensor
import Data.Tensor.Compile (ExprHashed, DimWrap(..), TensorWrap(..), attrDim)
import Comp
import Hmatrix
import Gen

--
--  The Generation of Expr is slow, so better to restrict the max size in test data generation.
--
-- stack build tensor-expression --test --test-arguments --qc-max-size=25
--

mkZ :: Dimension d => d -> IO (Tensor d Float)
mkZ = newTensor

mkV :: Dimension d => d -> Gen (PV.Vector Float)
mkV d = let sf = fromIntegral <$> (arbitrary :: Gen Int)
        in PV.fromList <$> vectorOf (size d) sf

isclose a b
  | a == b       = True
  | isInfinite a = False
  | isInfinite b = False
  | otherwise    = let diff = abs (a-b)
                       rel  = 0.0001
                   in diff <= abs (rel * a) || diff <= abs (rel * b)


eq :: (Dimension d, Element a, RealFloat a) => Tensor d a -> Tensor d a -> IO Property
eq t1 t2 = do
  d1 <- PV.unsafeFreeze (_tdat t1)
  d2 <- PV.unsafeFreeze (_tdat t2)
  let cc = PV.zipWith isclose d1 d2
      pr = do putStrLn $ show d1
              putStrLn $ show d2
  return $ whenFail pr $ _tdim t1 == _tdim t2 && PV.and cc

main = hspec $ do
  describe "vec + vec" $ do
    it "zero + vec1 = vec1" $ do
      let d = D1 5
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- packTensor d v
          t2 <- eval' $ I zr :.+ I t1
          eq t1 t2
    it "vec1 + zero = vec1" $ do
      let d = D1 5
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- packTensor d v
          t2 <- eval' $ I t1 :.+ I zr
          eq t1 t2
    it "vec1 + vec2 = vec3" $ do
      let d = D1 7
      forAll (liftM2 (,) (mkV d) (mkV d)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d v1
          t2 <- packTensor d v2
          t3 <- packTensor d $ PV.zipWith (+) v1 v2
          t4 <- eval' $ I t1 :.+ I t2
          eq t3 t4
  describe "vec * vec" $ do
    it "zero * vec1 = zero" $ do
      let d = D1 3
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- packTensor d v
          t2 <- eval' $ I zr :.* I t1
          eq zr t2
    it "vec1 * zero = zero" $ do
      let d = D1 3
      forAll (mkV d) $
        \v -> ioProperty $ do
          zr <- mkZ d
          t1 <- I <$> packTensor d v
          t2 <- eval' $ I zr :.* t1
          eq zr t2
    it "vec1 * vec2 = vec3" $ do
      let d = D1 7
      forAll (liftM2 (,) (mkV d) (mkV d)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d v1
          t2 <- packTensor d v2
          t3 <- packTensor d $ PV.zipWith (*) v1 v2
          t4 <- eval' $ I t1 :.* I t2
          eq t3 t4
  describe "vec * matrix" $ do
    it "vec1 * ident = vec1" $ do
      let s = 3
      forAll (mkV (D1 s)) $ do
        \v1 -> ioProperty $ do
          t1 <- packTensor (D1 s) v1
          t2 <- packTensor (D2 s s) $ hm2v (ident s)
          t3 <- eval' $ I t1 :<# I t2
          eq t3 t1
    it "vec1 * matrix = vec2" $ do
      let d1 = D1 2
          d2 = D2 2 6
          d3 = D1 6
      forAll (liftM2 (,) (mkV d1) (mkV d2)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d1 v1
          t2 <- packTensor d2 v2
          t3 <- packTensor d3 $ hv2v (v2hv d1 v1 <# v2hm d2 v2)
          t4 <- eval' $ I t1 :<# I t2
          eq t3 t4
  describe "matrix * vec" $ do
    it "ident * vec1 = vec1" $ do
      let s = 3
      forAll (mkV (D1 s)) $ do
        \v1 -> ioProperty $ do
          t1 <- packTensor (D2 s s) $ hm2v (ident s)
          t2 <- packTensor (D1 s) v1
          t3 <- eval' $ I t1 :#> I t2
          eq t2 t3
    it "matrix * vec1 = vec2" $ do
      let d1 = D2 6 2
          d2 = D1 2
          d3 = D1 6
      forAll (liftM2 (,) (mkV d1) (mkV d2)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d1 v1
          t2 <- packTensor d2 v2
          t3 <- packTensor d3 $ hv2v (v2hm d1 v1 #> v2hv d2 v2)
          t4 <- eval' $ I t1 :#> I t2
          eq t3 t4
  describe "matrix * matrix" $ do
    it "ident %# matrix1 = matrix1" $ do
      let s = 8
      forAll (mkV (D2 s s)) $ do
        \v1 -> ioProperty $ do
          t1 <- packTensor (D2 s s) $ hm2v (ident s)
          t2 <- packTensor (D2 s s) v1
          t3 <- eval' $ I t1 :%# I t2
          eq t2 t3
    it "matrix1 %# ident = matrix1^T" $ do
      let s = 8
          d = D2 s s
      forAll (mkV d) $ do
        \v1 -> ioProperty $ do
          t1 <- packTensor d v1
          t2 <- packTensor d $ hm2v (ident s)
          t3 <- packTensor d $ hm2v (tr' $ v2hm d v1)
          t4 <- eval' $ I t1 :%# I t2
          eq t3 t4
    it "matrix1 %# matrix2 = matrix3" $ do
      let d1 = D2 8 5
          d2 = D2 3 5
          d3 = D2 3 8
      forAll (liftM2 (,) (mkV d1) (mkV d2)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d1 v1
          t2 <- packTensor d2 v2
          t3 <- packTensor d3 $ hm2v (v2hm d2 v2 <> tr' (v2hm d1 v1))
          t4 <- eval' $ I t1 :%# I t2
          eq t3 t4
  describe "cross product" $ do
    it "vec1 <> vec2" $ do
      let d1 = D1 3
          d2 = D1 5
          d3 = D2 3 5
      forAll (liftM2 (,) (mkV d1) (mkV d2)) $ do
        \(v1, v2) -> ioProperty $ do
          t1 <- packTensor d1 v1
          t2 <- packTensor d2 v2
          t3 <- eval' $ I t1 :<> I t2
          t4 <- packTensor d3 $ hm2v (v2hv d1 v1 `outer` (v2hv d2 v2))
          eq t3 t4
  describe "scaling" $ do
    it "scale vector" $ do
      let d = D1 9
      forAll (mkV d) $ \v1 ->
        forAll (choose (0,100 :: Int)) $ \s ->
          ioProperty $ do
            let f = fromIntegral s / 100
            t1 <- packTensor d v1
            t2 <- eval' $ S f (I t1)
            t3 <- packTensor d $ hv2v (scale f $ v2hv d v1)
            eq t2 t3
    it "scale matrix" $ do
      let d = D2 8 8
      forAll (mkV d) $ \v1 ->
        forAll (choose (0,100 :: Int)) $ \s ->
          ioProperty $ do
            let f = fromIntegral s / 100
            t1 <- packTensor d v1
            t2 <- eval' $ S f (I t1)
            t3 <- packTensor d $ hm2v (scale f $ v2hm d v1)
            eq t2 t3
  describe "optimization" $ do
    it "opt'ed 1D Expr computes right" $ property $ \ (e :: Expr D1 Float) -> ioProperty $ do
      t1 <- eval' e
      t2 <- eval  e
      eq t1 t2
    it "opt'ed 2D Expr computes right" $ property $ \ (e :: Expr D2 Float) -> ioProperty $ do
      t1 <- eval' e
      t2 <- eval  e
      eq t1 t2
  describe "CSE" $ do
    it "substitutes one common sub-expr" $ do
      forAll (arbitrary `suchThat` notVI) $ \e -> ioProperty $ do
        s  <- generate (resize 3 arbitrary)
        (e1, rc) <- insert_ce 2 s e
        let e2 = fst $ elimCommonExpr e1
        let (vs, ss) = unzip (diff_ce e2 e1)
        return $ 
          (rc >= 2) ==> not (null vs)                     -- at least one common sub-expr
                     && and (map (== head vs) (tail vs))  -- all subst'd var  are the same
                     && and (map (== s) ss)               -- all subst'd expr are the same
    it "preserve value" $ do
      forAll (arbitrary `suchThat` notVI) $ \e -> ioProperty $ do
        s  <- generate (resize 3 arbitrary)
        (e1, rc) <- insert_ce 2 s e
        let e2 = uncurry qualify $ elimCommonExpr e1
        t1 <- evalExprHashed e1
        t2 <- evalExprHashed e2
        case (t1, t2) of 
          (TensorWrap t1, TensorWrap t2) 
            | Just t2 <- cast t2 -> eq t1 t2
            | otherwise          -> return $ property False

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