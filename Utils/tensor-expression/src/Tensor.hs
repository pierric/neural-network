module Tensor(
  Dimension(..), Tensor(..), Expr(..), Var(..), Statement(..),
  D1(..), D2(..), D3(..),
  isAlloc, isBind, isStore, isStoreTo,
  compile, newTensor, tensor_eq,
) where

-- import Blas.Generic.Unsafe
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V
import Foreign.ForeignPtr (castForeignPtr)
import Control.Monad.State
import Control.Monad.Except
import Data.Data
import Text.Printf
import Text.PrettyPrint.Free (Pretty(..), Doc, fill, text, vcat, (<+>))

data Order     = RowMajor | ColumnMajor
data Transpose = Trans | NoTrans

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

data Tensor d a = Tensor {
  _tdim :: d,
  _tdat :: (V.IOVector a)
}

data Expr d a where
  I :: Tensor d a -> Expr d a
  -- S :: a -> Expr d a -> Expr d a
  -- A :: (a -> a) -> Expr d a -> Expr d a
  -- (:.*) :: Expr d a -> Expr d a -> Expr d a
  (:.+) :: Expr d a -> Expr d a -> Expr d a
  (:<#) :: Expr D1 a -> Expr D2 a -> Expr D1 a
  -- (:#>) :: Expr D2 a -> Expr D1 a -> Expr D1 a
  -- (:<>) :: Expr D2 a -> Expr D2 a -> Expr D2 a
  -- (:%#) :: Expr D1 a -> Expr D1 a -> Expr D2 a

data Var d a = Var {
  _vdim :: d,
  _vid  :: Int
} deriving (Typeable, Data)

data Statement where
  Alloc    :: (Dimension d, Typeable a, Show a) => Var d a -> Statement
  Bind     :: (Dimension d, Typeable a, Show a) => Var d a -> Tensor d a -> Statement
  Store    :: (Dimension d, Typeable a, Show a) => Var d a -> Tensor d a -> Statement
  -- BlasGERU :: Order -> Var D1 a -> Var D1 a -> a -> Var D2 a -> Statement
  -- BlasGEMM :: Order -> Transpose -> Var D2 a -> Transpose -> Var D2 a -> a -> a -> Var D2 a -> Statement
  BlasGEMV :: Show a => Order -> Transpose -> Var D2 a -> Var D1 a -> a -> a -> Var D1 a -> Statement
  DotAdd   :: (Dimension d, Show a) => Var d a -> Var d a -> Var d a -> Statement
  -- DotMul   :: Var d a -> Var d a -> Var d a -> Statement

isAlloc (Alloc _) = True
isAlloc _         = False

isBind (Bind _ _) = True
isBind _          = False

isStore (Store _ _) = True
isStore _           = False

isStoreTo vid (Store v _) | vid == _vid v = True
isStoreTo _ _                             = False

instance (Show d, Show a) => Show (Tensor d a) where
  show (Tensor d (V.MVector o v)) = printf "<tensor (%-8s): %s + %4d>" (show d) (show v) o
instance (Show d, Show a) => Show (Var d a) where
  show (Var d i) = printf "v%03d" i

instance Pretty Statement where
  pretty (Alloc v)   = fill 8 (text "Alloc") <+> text (printf "%s" (show v))
  pretty (Bind  v t) = fill 8 (text "Bind")  <+> text (printf "%s %s" (show v) (show t))
  pretty (Store v t) = fill 8 (text "Store") <+> text (printf "%s %s" (show v) (show t))
  pretty (BlasGEMV o t v1 v2 a b v3) = fill 8 (text "Gemv") <+> text (printf "%s %s %s" (show v1) (show v2) (show v3))
  pretty (DotAdd v1 v2 v3)           = fill 8 (text "Add")  <+> text (printf "%s %s %s" (show v1) (show v2) (show v3))

type CG = StateT CGState (ExceptT CGError IO)
data CGError = CGSizeMismatchedTensors
  deriving Show
type CGState = Int

newVar :: d -> CG (Var d a)
newVar d = do
  i <- get
  modify (+1)
  return $ Var d i

compile :: (Dimension d, Typeable a, Show a, Num a) => Expr d a -> CG ([Statement], Var d a)
compile (I t) = do
  v <- newVar (_tdim t)
  return ([Bind v t], v)
compile (a :.+ b) = do
  (s1, v1) <- compile a
  (s2, v2) <- compile b
  if _vdim v1 /= _vdim v2
    then throwError CGSizeMismatchedTensors
    else do
      v3 <- newVar (_vdim v1)
      return (s1 ++ s2 ++ [Alloc v3, DotAdd v1 v2 v3], v3)
compile (a :<# b) = do
  (s1, v1) <- compile a
  (s2, v2) <- compile b
  let d1@(D1 k)   = _vdim v1
      d2@(D2 m n) = _vdim v2
  if k /= m
    then throwError CGSizeMismatchedTensors
    else do
      v3 <- newVar (D1 n)
      return (s1 ++ s2 ++ [Alloc v3, BlasGEMV RowMajor Trans v2 v1 1 0 v3], v3)

newTensor :: (Dimension d, V.Storable a) => d -> IO (Tensor d a)
newTensor d = Tensor d <$> V.new (size d)

tensor_eq :: (Dimension d1, Dimension d2) => Tensor d1 a1 -> Tensor d2 a2 -> Bool
tensor_eq (Tensor d1 (V.MVector o1 p1)) (Tensor d2 (V.MVector o2 p2)) =
  size d1 == size d2 && o1 == o2 && p1 == castForeignPtr p2
