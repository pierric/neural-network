module Data.Tensor.Class where

import qualified Data.Vector.Storable as PV
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V
import Foreign.ForeignPtr (castForeignPtr)
import Data.Typeable (cast)
import Data.Data
import Blas.Generic.Safe (Numeric)
import Data.Tensor.SIMD (SIMDable)

data D1 = D1 {-# UNPACK #-}!Int
  deriving (Typeable, Data, Eq, Show)
data D2 = D2 {-# UNPACK #-}!Int {-# UNPACK #-}!Int
  deriving (Typeable, Data, Eq, Show)
data D3 = D3 {-# UNPACK #-}!Int {-# UNPACK #-}!Int {-# UNPACK #-}!Int
  deriving (Typeable, Data, Eq, Show)

class (Show a, Eq a, Typeable a) => Dimension a where
  size :: a -> Int

instance Dimension D1 where
  size (D1 a) = a
instance Dimension D2 where
  size (D2 a b) = a * b
instance Dimension D3 where
  size (D3 a b c) = a * b * c

class (Show a, Num a, Eq a, Typeable a, V.Storable a, SIMDable a, Numeric a) => Element a

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