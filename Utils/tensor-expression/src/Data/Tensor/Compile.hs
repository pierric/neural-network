{-# LANGUAGE ScopedTypeVariables #-}
module Data.Tensor.Compile(
  Expr(..), dim,
  Var(..), Statement(..),
  isAlloc, isBind, isBindToTensor, isStore, isStoreTo,
  isGEMV, isGERU, isGEMM,
  isDotAdd, isDotAddTo, isDotSca, isDotScaTo,
  CGState, EvalE,
  runCG, compile, substitute, execute,
) where

import Data.Tensor.Class
import Data.Data
import Control.Monad.State
import Control.Monad.Except
import Text.Printf
import Text.PrettyPrint.Free (Pretty(..), Doc, fill, text, hsep, vcat, (<+>))
import qualified Data.Map.Strict as M
import Blas.Generic.Safe
import Blas.Primitive.Types (Order(..), Transpose(..))
import Data.Tensor.SIMD
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V

import Foreign.Marshal.Array
dump ptr sz = do
  fs <- peekArray sz ptr
  print fs

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

dim :: Expr d a -> d
dim (I t)     = _tdim t
dim (a :.* b) = dim a
dim (a :.+ b) = dim b
dim (a :<# b) = let D2 _ c = dim b in D1 c
dim (a :#> b) = let D2 r _ = dim a in D1 r
dim (a :%# b) = let (D2 r1 _, D2 r2 _) = (dim a, dim b) in D2 r2 r1
dim (a :<> b) = let (D1 r, D1 c) = (dim a, dim b) in D2 r c

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
isBindToTensor t1 s | Bind _ t2 <- s, eqTensor t1 t2 = True
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

instance Show Statement where
  show = show . pretty

type CG = StateT CGState (ExceptT CGError IO)
data CGError = CGSizeMismatchedTensors
  deriving (Eq, Show)
type CGState = Int

runCG :: CGState -> CG a -> IO (Either CGError (a, CGState))
runCG cg act = runExceptT (runStateT act cg )

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
  if n /= k
    then throwError CGSizeMismatchedTensors
    else do
      v3 <- newVar (D1 m)
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
      v3 <- newVar (D2 u m)
      return (s1 ++ s2 ++ [Alloc v3, BlasGEMM ColMajor Trans v1 NoTrans v2 1 0 v3], v3)

data TensorWrap where
  TensorWrap :: (Dimension d, Element a) => Tensor d a -> TensorWrap
type EvalS = M.Map Int TensorWrap
type EvalM = ExceptT EvalE (StateT EvalS IO)
data EvalE = Fail Statement
  deriving (Show)

execute :: [Statement] -> IO (Either EvalE ())
execute ss = evalStateT (runExceptT $ eval ss) M.empty
  where
    eval [] = return ()
    eval (s:ss) = do
      case s of
        Alloc (v :: Var d a) -> do
          m <- get
          if M.member (_vid v) m
            then throwError $ Fail s
            else do
              (t :: Tensor d a) <- liftIO $ newTensor (_vdim v)
              put $ M.insert (_vid v) (TensorWrap t) m
        Bind  v t -> do
          m <- get
          if M.member (_vid v) m
            then throwError $ Fail s
            else put $ M.insert (_vid v) (TensorWrap t) m
        Store v t -> do
          m <- get
          case M.lookup (_vid v) m of
            Just tw | TensorWrap tt <- tw
                    , size (_tdim tt) == size (_vdim v)
                    , Just tt' <- cast tt
                    -> liftIO $ copyTensor tt' t
            _       -> throwError $ Fail s
        Copy  v w -> do
          m <- get
          let vt = M.lookup (_vid v) m
              wt = M.lookup (_vid w) m
          case (vt, wt) of
            _ | Just (TensorWrap vt) <- vt
              , Just (TensorWrap wt) <- wt
              , Just vt' <- cast vt
              -> liftIO $ copyTensor vt' wt
            _ -> throwError $ Fail s
        BlasGEMV o t v w a b u ->
          dotop2 (throwError $ Fail s) v w u (\tx ty tz -> liftIO $
            V.unsafeWith (_tdat tx) (\px ->
            V.unsafeWith (_tdat ty) (\py ->
            V.unsafeWith (_tdat tz) (\pz ->
              let (D2 r c) = _vdim v
                  (m,n,lda) = case o of
                                RowMajor -> (r,c,c)
                                ColMajor -> (c,r,c)
              in gemv o t m n a px lda py 1 b pz 1))))
        BlasGERU o v w a u ->
          dotop2 (throwError $ Fail s) v w u (\tx ty tz -> liftIO $
            V.unsafeWith (_tdat tx) (\px ->
            V.unsafeWith (_tdat ty) (\py ->
            V.unsafeWith (_tdat tz) (\pz ->
              let D1 r = _vdim v
                  D1 c = _vdim w
                  (m,n,lda) = case o of
                                RowMajor -> (r, c, r)
                                ColMajor -> (c, r, c)
              in geru o m n a px 1 py 1 pz lda))))
        BlasGEMM o t1 v1 t2 v2 a b w ->
          dotop2 (throwError $ Fail s) v1 v2 w (\tx ty tz -> liftIO $
            V.unsafeWith (_tdat tx) (\px ->
            V.unsafeWith (_tdat ty) (\py ->
            V.unsafeWith (_tdat tz) (\pz ->
              let D2 r1 c1 = _vdim v1
                  D2 r2 c2 = _vdim v2
                  (m,k,lda,n,ldb) = case (o,t1,t2) of
                                      (RowMajor,   Trans,   Trans) -> (c1, r1, r1, r2, r2)
                                      (RowMajor,   Trans, NoTrans) -> (c1, r1, r1, c2, r2)
                                      (RowMajor, NoTrans,   Trans) -> (r1, c1, r1, r2, r2)
                                      (RowMajor, NoTrans, NoTrans) -> (r1, c1, r1, c2, r2)
                                      (ColMajor,   Trans,   Trans) -> (r1, c1, c1, c2, c2)
                                      (ColMajor,   Trans, NoTrans) -> (r1, c1, c1, r2, c2)
                                      (ColMajor, NoTrans,   Trans) -> (c1, r1, c1, c2, c2)
                                      (ColMajor, NoTrans, NoTrans) -> (c1, r1, c1, r2, c2)
              in gemm o t1 t2 m n k a px lda py ldb b pz m))))
        DotAdd v w u -> dotop2 (throwError $ Fail s) v w u (\tv tw tu -> liftIO $
                          hadamard plus  (_tdat tu) (_tdat tv) (_tdat tw))
        DotMul v w u -> dotop2 (throwError $ Fail s) v w u (\tv tw tu -> liftIO $
                          hadamard times (_tdat tu) (_tdat tv) (_tdat tw))
        DotSca a v w -> dotop1 (throwError $ Fail s) v w (\tv tw -> liftIO $
                          foreach (times (konst a)) (_tdat tw) (_tdat tv))
      eval ss

    dotop1 :: forall d1 d2 a b. (Dimension d1, Dimension d2, Element a) =>
              EvalM b -> Var d1 a -> Var d2 a -> (Tensor d1 a -> Tensor d2 a -> EvalM b) -> EvalM b
    dotop1 e v w o = do
      m <- get
      let vt = M.lookup (_vid v) m
          wt = M.lookup (_vid w) m
      case (vt, wt) of
        _ | Just (TensorWrap vt) <- vt
          , Just (TensorWrap wt) <- wt
          , Just (vt' :: Tensor d1 a) <- cast vt
          , Just (wt' :: Tensor d2 a) <- cast wt
          -> o vt' wt'
        _ -> e

    dotop2 :: forall d1 d2 d3 a b. (Dimension d1, Dimension d2, Dimension d3, Element a) =>
              EvalM b -> Var d1 a -> Var d2 a -> Var d3 a -> (Tensor d1 a -> Tensor d2 a -> Tensor d3 a -> EvalM b) -> EvalM b
    dotop2 e u v w o = do
      m <- get
      let ut = M.lookup (_vid u) m
          vt = M.lookup (_vid v) m
          wt = M.lookup (_vid w) m
      case (ut, vt, wt) of
        _ | Just (TensorWrap ut) <- ut
          , Just (TensorWrap vt) <- vt
          , Just (TensorWrap wt) <- wt
          , Just (ut' :: Tensor d1 a) <- cast ut
          , Just (vt' :: Tensor d2 a) <- cast vt
          , Just (wt' :: Tensor d3 a) <- cast wt
          -> o ut' vt' wt'
        _ -> e
