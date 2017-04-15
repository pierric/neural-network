module Data.Tensor.Class where

import qualified Data.Vector.Storable as PV
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V
import Text.Printf
import Foreign.ForeignPtr (castForeignPtr)
import Data.Typeable (cast)
import Data.Data
import Data.Hashable
import Blas.Generic.Safe (Numeric)
import Control.Monad.State.Strict
import Control.Monad.Except
import System.IO.Unsafe (unsafeDupablePerformIO)
import Foreign.Ptr (Ptr)
import Data.Tensor.SIMD (SIMDable)

data D1 = D1 {-# UNPACK #-}!Int
  deriving (Typeable, Data, Eq, Show)
data D2 = D2 {-# UNPACK #-}!Int {-# UNPACK #-}!Int
  deriving (Typeable, Data, Eq, Show)
data D3 = D3 {-# UNPACK #-}!Int {-# UNPACK #-}!Int {-# UNPACK #-}!Int
  deriving (Typeable, Data, Eq, Show)

class (Show a, Eq a, Typeable a, Hashable a) => Dimension a where
  size :: a -> Int

instance Dimension D1 where
  size (D1 a) = a
instance Dimension D2 where
  size (D2 a b) = a * b
instance Dimension D3 where
  size (D3 a b c) = a * b * c

class (Show a, Num a, Eq a, Typeable a, V.Storable a, Hashable a, SIMDable a, Numeric a) => Element a

instance Element Float

data Tensor d a = Tensor {
  _tdim :: d,
  _tdat :: (V.IOVector a)
}

newTensor :: (Dimension d, Element a) => d -> IO (Tensor d a)
newTensor d = Tensor d <$> V.new (size d)

copyTensor :: (Dimension d, Element a) => Tensor d a -> Tensor d a -> IO ()
copyTensor (Tensor d1 a) (Tensor d2 b) = V.copy b a

packTensor :: (Dimension d, Element a) => d -> PV.Vector a -> IO (Tensor d a)
packTensor d v = if PV.length v < size d then error "cannot pack tensor" else do
  v <- PV.unsafeThaw v
  return $ Tensor d v

eqTensor :: (Dimension d1, Dimension d2) => Tensor d1 a1 -> Tensor d2 a2 -> Bool
eqTensor (Tensor d1 (V.MVector o1 p1)) (Tensor d2 (V.MVector o2 p2)) =
  size d1 == size d2 && o1 == o2 && p1 == castForeignPtr p2

data Expr d a where
  I :: Tensor d a -> Expr d a
  S :: a -> Expr d a -> Expr d a
  -- A :: (a -> a) -> Expr d a -> Expr d a
  (:.*) :: Expr d a -> Expr d a -> Expr d a
  (:.+) :: Expr d a -> Expr d a -> Expr d a
  (:<#) :: Expr D1 a -> Expr D2 a -> Expr D1 a
  (:#>) :: Expr D2 a -> Expr D1 a -> Expr D1 a
  (:%#) :: Expr D2 a -> Expr D2 a -> Expr D2 a
  (:<>) :: Expr D1 a -> Expr D1 a -> Expr D2 a

dim :: Expr d a -> d
dim (I t)     = _tdim t
dim (S _ a)   = dim a
dim (a :.* b) = dim a
dim (a :.+ b) = dim b
dim (a :<# b) = let D2 _ c = dim b in D1 c
dim (a :#> b) = let D2 r _ = dim a in D1 r
dim (a :%# b) = let (D2 r1 _, D2 r2 _) = (dim a, dim b) in D2 r2 r1
dim (a :<> b) = let (D1 r, D1 c) = (dim a, dim b) in D2 r c

instance (Show d, Show a) => Show (Tensor d a) where
  show (Tensor d (V.MVector o v)) = printf "<tensor (%-8s): %s + %4d>" (show d) (show v) o

deriving instance (Show d, Show a) => Show (Expr d a)

type VarId = Int

instance Hashable D1 where
  hashWithSalt s (D1 a) = s `hashWithSalt` a

instance Hashable D2 where
  hashWithSalt s (D2 a b) = s `hashWithSalt` a `hashWithSalt` b

instance Hashable D3 where
  hashWithSalt s (D3 a b c) = s `hashWithSalt` a `hashWithSalt` b `hashWithSalt` c

instance (Hashable d, Hashable a, V.Storable a) => Hashable (Tensor d a) where
  hashWithSalt s (Tensor d v) = s `hashWithSalt` d `hashWithSalt` (unsafeDupablePerformIO $ V.unsafeWith v return)
--
-- instance (Hashable a, V.Storable a) => Hashable (Expr D1 a) where
--   hashWithSalt s (I t)     = s `hashWithSalt` d1Expr `hashWithSalt` iExpr  `hashWithSalt` t
--   hashWithSalt s (S f e)   = s `hashWithSalt` d1Expr `hashWithSalt` sExpr  `hashWithSalt` f `hashWithSalt` e
--   hashWithSalt s (a :.* b) = s `hashWithSalt` d1Expr `hashWithSalt` dmExpr `hashWithSalt` a `hashWithSalt` b
--   hashWithSalt s (a :.+ b) = s `hashWithSalt` d1Expr `hashWithSalt` daExpr `hashWithSalt` a `hashWithSalt` b
--   hashWithSalt s (a :<# b) = s `hashWithSalt` d1Expr `hashWithSalt` vmExpr `hashWithSalt` a `hashWithSalt` b
--   hashWithSalt s (a :#> b) = s `hashWithSalt` d1Expr `hashWithSalt` mvExpr `hashWithSalt` a `hashWithSalt` b
--
-- instance (Hashable a, V.Storable a) => Hashable (Expr D2 a) where
--   hashWithSalt s (I t)     = s `hashWithSalt` d2Expr `hashWithSalt` iExpr   `hashWithSalt` t
--   hashWithSalt s (S f e)   = s `hashWithSalt` d2Expr `hashWithSalt` sExpr   `hashWithSalt` f `hashWithSalt` e
--   hashWithSalt s (a :.* b) = s `hashWithSalt` d2Expr `hashWithSalt` dmExpr  `hashWithSalt` a `hashWithSalt` b
--   hashWithSalt s (a :.+ b) = s `hashWithSalt` d2Expr `hashWithSalt` daExpr  `hashWithSalt` a `hashWithSalt` b
--   hashWithSalt s (a :%# b) = s `hashWithSalt` d2Expr `hashWithSalt` mtmExpr `hashWithSalt` a `hashWithSalt` b
--   hashWithSalt s (a :<> b) = s `hashWithSalt` d2Expr `hashWithSalt` ovvExpr `hashWithSalt` a `hashWithSalt` b
--
-- d1Expr, d2Expr :: Int
-- d1Expr = 1
-- d2Expr = 2
-- iExpr, sExpr, dmExpr, daExpr, vmExpr, mvExpr, mtmExpr, ovvExpr :: Int
-- iExpr  = 0x100
-- sExpr  = 0x101
-- dmExpr = 0x102
-- daExpr = 0x103
-- vmExpr = 0x104
-- mvExpr = 0x105
-- mtmExpr= 0x106
-- ovvExpr= 0x107
