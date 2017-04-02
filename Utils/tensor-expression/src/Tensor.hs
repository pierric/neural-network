module Tensor(
  Dimension(..), Tensor(..), Expr(..), Var(..), Statement(..),
  D1(..), D2(..), D3(..),
  isAlloc, isBind, isBindToTensor, isStore, isStoreTo,
  isGEMV, isGERU, isGEMM,
  isDotAdd, isDotAddTo, isDotSca, isDotScaTo,
  newTensor, tensor_eq,
  CGState,
  compile, substitute,
) where

-- import Blas.Generic.Unsafe
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V
import Foreign.ForeignPtr (castForeignPtr)
import Data.Typeable (cast)
import Control.Monad.State
import Control.Monad.Except
import Data.Data
import qualified Data.Map.Strict as M
import Text.Printf
import Text.PrettyPrint.Free (Pretty(..), Doc, fill, text, hsep, vcat, (<+>))

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

class (Show a, Num a, Eq a, Typeable a, V.Storable a) => Element a

instance Element Float

data Tensor d a = Tensor {
  _tdim :: d,
  _tdat :: (V.IOVector a)
}

data Expr d a where
  I :: Tensor d a -> Expr d a
  -- S :: a -> Expr d a -> Expr d a
  -- A :: (a -> a) -> Expr d a -> Expr d a
  (:.*) :: Expr d a -> Expr d a -> Expr d a
  (:.+) :: Expr d a -> Expr d a -> Expr d a
  (:<#) :: Expr D1 a -> Expr D2 a -> Expr D1 a
  (:#>) :: Expr D2 a -> Expr D1 a -> Expr D1 a
  (:%#) :: Expr D2 a -> Expr D2 a -> Expr D2 a
  (:<>) :: Expr D1 a -> Expr D1 a -> Expr D2 a

data Var d a = Var {
  _vdim :: d,
  _vid  :: Int
} deriving (Typeable, Data, Eq)

data Statement where
  Alloc    :: (Dimension d, Element a) => Var d a -> Statement
  Bind     :: (Dimension d, Element a) => Var d a -> Tensor d a -> Statement
  Store    :: (Dimension d, Element a) => Var d a -> Tensor d a -> Statement
  Copy     :: (Dimension d, Element a) => Var d a -> Var d a -> Statement
  BlasGEMV :: Element a => Order -> Transpose -> Var D2 a -> Var D1 a -> a -> a -> Var D1 a -> Statement
  BlasGERU :: Element a => Order -> Var D1 a -> Var D1 a -> a -> Var D2 a -> Statement
  BlasGEMM :: Element a => Order -> Transpose -> Var D2 a -> Transpose -> Var D2 a -> a -> a -> Var D2 a -> Statement
  DotAdd   :: (Dimension d, Element a) => Var d a -> Var d a -> Var d a -> Statement
  DotMul   :: (Dimension d, Element a) => Var d a -> Var d a -> Var d a -> Statement
  DotSca   :: (Dimension d, Element a) => a -> Var d a -> Var d a -> Statement

isAlloc (Alloc{}) = True
isAlloc _         = False

isBind (Bind{}) = True
isBind _          = False
isBindToTensor t1 s | Bind _ t2 <- s, tensor_eq t1 t2 = True
                    | otherwise = False

isStore (Store{}) = True
isStore _           = False
isStoreTo vid s | Store v _ <-s, vid == _vid v = True
                | otherwise = False

isGEMV (BlasGEMV{}) = True
isGEMV _            = False

isGERU (BlasGERU{}) = True
isGERU _            = False

isGEMM (BlasGEMM{}) = True
isGEMM _            = False

isDotAdd (DotAdd{}) = True
isDotAdd _          = False
isDotAddTo vid s | DotAdd v1 v2 _ <- s, vid == _vid v1 || vid == _vid v2 = True
                 | otherwise = False

isDotSca (DotSca{}) = True
isDotSca _          = False
isDotScaTo vid s | DotSca a v1 v2 <- s, vid == _vid v1 = True
                 | otherwise = False

substitute :: (Dimension d, Element a) => Var d a -> Var d a -> [Statement] -> [Statement]
substitute v1 v2 st = if v1 == v2 then st else map subst st
  where
    subst (Store v3 t)
      | Just v1 == cast v3 = Store (v3{_vid = _vid v2}) t
    subst (BlasGEMV o t va vb a b vc)
      | Just v1 == cast va = BlasGEMV o t (va{_vid = _vid v2}) vb a b vc
      | Just v1 == cast vb = BlasGEMV o t va (vb{_vid = _vid v2}) a b vc
      | Just v1 == cast vc = BlasGEMV o t va vb a b (vc{_vid = _vid v2})
    subst (DotAdd va vb vc)
      | Just v1 == cast va = DotAdd (va{_vid = _vid v2}) vb vc
      | Just v1 == cast vb = DotAdd va (vb{_vid = _vid v2}) vc
      | Just v1 == cast vc = DotAdd va vb (vc{_vid = _vid v2})

instance (Dimension d, Element a) => Show (Tensor d a) where
  show (Tensor d (V.MVector o v)) = printf "<tensor (%-8s): %s + %4d>" (show d) (show v) o
instance (Dimension d, Element a) => Show (Var d a) where
  show (Var d i) = printf "v%03d" i

instance Pretty Statement where
  pretty (Alloc v)   = fill 8 (text "Alloc") <+> text (printf "%s" (show v))
  pretty (Bind  v t) = fill 8 (text "Bind")  <+> text (printf "%s %s" (show v) (show t))
  pretty (Store v t) = fill 8 (text "Store") <+> text (printf "%s %s" (show v) (show t))
  pretty (Copy v1 v2)= fill 8 (text "Copy")  <+> text (printf "%s %s" (show v1) (show v2))
  pretty (BlasGEMV o t v1 v2 a b v3)     = fill 8 (text "Gemv") <+> hsep (map text [show v1, show v2, show v3, show a, show b])
  pretty (BlasGERU o v1 v2 a v3)         = fill 8 (text "Geru") <+> hsep (map text [show v1, show v2, show v3, show a])
  pretty (BlasGEMM o t1 v1 t2 v2 a b v3) = fill 8 (text "Gemm") <+> hsep (map text [show v1, show v2, show v3, show a, show b])
  pretty (DotAdd v1 v2 v3)               = fill 8 (text "Add")  <+> hsep (map text [show v1, show v2, show v3])
  pretty (DotMul v1 v2 v3)               = fill 8 (text "Mul")  <+> hsep (map text [show v1, show v2, show v3])
  pretty (DotSca a  v1 v2)               = fill 8 (text "Sca")  <+> hsep (map text [show a , show v1, show v2])

type CG = StateT CGState (ExceptT CGError IO)
data CGError = CGSizeMismatchedTensors
  deriving Show
type CGState = Int

newVar :: MonadState CGState m => d -> m (Var d a)
newVar d = do
  i <- get
  modify (+1)
  return $ Var d i

compile :: (Dimension d, Element a) => Expr d a -> CG ([Statement], Var d a)
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
compile (a :.* b) = do
  (s1, v1) <- compile a
  (s2, v2) <- compile b
  if _vdim v1 /= _vdim v2
    then throwError CGSizeMismatchedTensors
    else do
      v3 <- newVar (_vdim v1)
      return (s1 ++ s2 ++ [Alloc v3, DotMul v1 v2 v3], v3)
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
compile (a :#> b) = do
  (s1, v1) <- compile a
  (s2, v2) <- compile b
  let d1@(D2 m n) = _vdim v1
      d2@(D1 k)   = _vdim v2
  if k /= m
    then throwError CGSizeMismatchedTensors
    else do
      v3 <- newVar (D1 n)
      return (s1 ++ s2 ++ [Alloc v3, BlasGEMV RowMajor NoTrans v1 v2 1 0 v3], v3)
compile (a :<> b) = do
  (s1, v1) <- compile a
  (s2, v2) <- compile b
  let d1@(D1 m) = _vdim v1
      d2@(D1 n) = _vdim v2
  v3 <- newVar (D2 m n)
  return (s1 ++ s2 ++ [Alloc v3, BlasGERU RowMajor v1 v2 1 v3], v3)
compile (a :%# b) = do
  (s1, v1) <- compile a
  (s2, v2) <- compile b
  let d1@(D2 m n) = _vdim v1
      d2@(D2 u v) = _vdim v2
  if n /= v
    then throwError CGSizeMismatchedTensors
    else do
      v3 <- newVar (D2 m n)
      return (s1 ++ s2 ++ [Alloc v3, BlasGEMM ColumnMajor Trans v1 NoTrans v2 1 0 v3], v3)

data TensorWrap where
  TensorWrap :: (Dimension d, Element a) => Tensor d a -> TensorWrap
type EvalS = M.Map Int TensorWrap
type EvalM = ExceptT EvalE (StateT EvalS IO)
data EvalE = CannotAlloc | CannotBind | CannotStore | CannotCopy

evaluate :: [Statement] -> IO (Either EvalE ())
evaluate ss = evalStateT (runExceptT $ eval ss) M.empty
  where
    eval [] = return ()
    eval (s:ss) = do
      case s of
        Alloc v -> do
          m <- get
          if M.member (_vid v) m
            then throwError CannotAlloc
            else do
              let newTensor' :: (Dimension d, Element a) => Var d a -> IO (Tensor d a)
                  newTensor' v = newTensor (_vdim v)
              t <- liftIO $ newTensor' v
              put $ M.insert (_vid v) (TensorWrap t) m
        Bind  v t -> do
          m <- get
          if M.member (_vid v) m
            then throwError CannotBind
            else put $ M.insert (_vid v) (TensorWrap t) m
        Store v t -> do
          m <- get
          case M.lookup (_vid v) m of
            Just tw | TensorWrap tt <- tw
                    , size (_tdim tt) == size (_vdim v)
                    , Just tt' <- cast tt
                    -> liftIO $ copyTensor tt' t
            _       -> throwError CannotStore
        Copy  v w -> do
          m <- get
          let vt = M.lookup (_vid v) m
              wt = M.lookup (_vid w) m
          case (vt, wt) of
            _ | Just (TensorWrap vt) <- vt
              , Just (TensorWrap wt) <- wt
              , Just vt' <- cast vt
              -> liftIO $ copyTensor vt' wt
            _ -> throwError CannotCopy
        BlasGEMV o t v w a b u -> undefined
        BlasGERU o v w a u -> undefined
        BlasGEMM o t1 v1 t2 v2 a b w -> undefined
        DotAdd v w u -> undefined
        DotMul v w u -> undefined
        DotSca a v w -> undefined
      eval ss

newTensor :: (Dimension d, Element a) => d -> IO (Tensor d a)
newTensor d = Tensor d <$> V.new (size d)

copyTensor :: (Dimension d, Element a) => Tensor d a -> Tensor d a -> IO ()
copyTensor (Tensor d1 a) (Tensor d2 b) = V.copy b a

tensor_eq :: (Dimension d1, Dimension d2) => Tensor d1 a1 -> Tensor d2 a2 -> Bool
tensor_eq (Tensor d1 (V.MVector o1 p1)) (Tensor d2 (V.MVector o2 p2)) =
  size d1 == size d2 && o1 == o2 && p1 == castForeignPtr p2
