{-# LANGUAGE FlexibleInstances, GADTs #-}
module Gen where

import Test.QuickCheck
import Data.Maybe
import qualified Data.Vector.Storable as PV
import Control.Monad
import Control.Monad.State
import Control.Monad.Reader
import Control.Monad.Trans.Maybe
import System.IO.Unsafe (unsafePerformIO)
import Data.Tensor

esize :: Expr d a -> Int
esize (I _)   = 1
esize (S _ _) = 1
esize (a :.* b) = 1 + esize a + esize b
esize (a :.+ b) = 1 + esize a + esize b
esize (a :<# b) = 1 + esize a + esize b
esize (a :#> b) = 1 + esize a + esize b
esize (a :%# b) = 1 + esize a + esize b
esize (a :<> b) = 1 + esize a + esize b

(??) = flip

instance (Arbitrary a, Element a) => Arbitrary (Expr D1 a) where
  arbitrary = sized (\es -> do a <- genSmallDimension
                               liftM fromJust (runGenM es (genExp1D (D1 a))))
instance (Arbitrary a, Element a) => Arbitrary (Expr D2 a) where
  arbitrary = sized (\es -> do a <- genSmallDimension
                               b <- genSmallDimension
                               liftM fromJust (runGenM es (genExp2D (D2 a b))))

data Op = Op_I
        | Op_S
        | Op_Dot_Add
        | Op_Dot_Mul
        | Op_VM
        | Op_MV
        | OP_MM_Spec
        | OP_V_Cross
  deriving (Bounded, Enum)

type GenM = ReaderT (Int,Int) (StateT Int (MaybeT Gen))

runGenM es = let es' = fromIntegral es
                 rng = (ceiling (es' * 0.8), floor (es' * 1.2))
             in runMaybeT . (evalStateT ?? 0) . (runReaderT ?? rng)
liftGenM = lift . lift . lift

check = do
  (_, maxsize) <- ask
  cursize <- get
  if cursize > maxsize then mzero else put (cursize+1)

uGenExp1D :: (Arbitrary a, Element a) => D1 -> GenM (Expr D1 a)
uGenExp1D d@(D1 a) = do
  op <- liftGenM $ elements [Op_I, Op_S, Op_Dot_Add, Op_Dot_Mul, Op_VM, Op_MV]
  check
  case op of
    Op_I -> I <$> liftGenM (genTensor d)
    Op_S -> S <$> liftGenM genSmallFraction <*> uGenExp1D d
    Op_Dot_Add -> pure (:.+) <*> uGenExp1D d <*> uGenExp1D d
    Op_Dot_Mul -> pure (:.*) <*> uGenExp1D d <*> uGenExp1D d
    Op_VM -> do b <- liftGenM genSmallDimension
                pure (:<#) <*> uGenExp1D (D1 b) <*> uGenExp2D (D2 b a)
    Op_MV -> do b <- liftGenM genSmallDimension
                pure (:#>) <*> uGenExp2D (D2 a b) <*> uGenExp1D (D1 b)

lGenExp1D :: (Arbitrary a, Element a) => D1 -> GenM (Expr D1 a)
lGenExp1D d = do
  e <- uGenExp1D d
  (minsize, _) <- ask
  if esize e < minsize then mzero else return e

genExp1D :: (Arbitrary a, Element a) => D1 -> GenM (Expr D1 a)
genExp1D d = lGenExp1D d `mplus` genExp1D d

uGenExp2D :: (Arbitrary a, Element a) => D2 -> GenM (Expr D2 a)
uGenExp2D d@(D2 a b) = do
  op <- liftGenM $ elements [Op_I, Op_S, Op_Dot_Add, Op_Dot_Mul, OP_MM_Spec, OP_V_Cross]
  check
  case op of
    Op_I -> I <$> liftGenM (genTensor d)
    Op_S -> S <$> liftGenM genSmallFraction <*> uGenExp2D d
    Op_Dot_Add -> pure (:.+) <*> uGenExp2D d <*> uGenExp2D d
    Op_Dot_Mul -> pure (:.*) <*> uGenExp2D d <*> uGenExp2D d
    OP_MM_Spec -> do c <- liftGenM genSmallDimension
                     pure (:%#) <*> uGenExp2D (D2 b c)<*> uGenExp2D (D2 a c)
    OP_V_Cross -> pure (:<>) <*> uGenExp1D (D1 a) <*> uGenExp1D (D1 b)

lGenExp2D :: (Arbitrary a, Element a) => D2 -> GenM (Expr D2 a)
lGenExp2D d = do
  e <- uGenExp2D d
  (minsize, _) <- ask
  if esize e < minsize then mzero else return e

genExp2D :: (Arbitrary a, Element a) => D2 -> GenM (Expr D2 a)
genExp2D d = lGenExp2D d `mplus` genExp2D d

genTensor :: (Dimension d, Element a, Arbitrary a) => d -> Gen (Tensor d a)
genTensor d = do vec <- PV.fromList <$> vectorOf (size d) genSmallFraction
                 return $! unsafePerformIO (packTensor d vec)

genSmallDimension :: Gen Int
genSmallDimension = getPositive <$> resize 40 arbitrary

genSmallFraction :: (Fractional a, Arbitrary a) => Gen a
genSmallFraction  = resize 5 arbitrary
