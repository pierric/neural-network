{-# LANGUAGE ScopedTypeVariables, Rank2Types #-}
module Data.Tensor.Execute where

import Data.Tensor.Class
import qualified Data.Tensor.Compile as E
import Data.Data
import Control.Monad.State.Strict
import Control.Monad.Except
import Text.Printf
import Text.PrettyPrint.Free (Pretty(..), Doc, fill, text, hsep, vcat, (<+>))
import qualified Data.Map.Strict as M
import Blas.Generic.Safe
import Blas.Primitive.Types (Order(..), Transpose(..))
import Data.Tensor.SIMD
import qualified Data.Vector.Storable.Mutable  as V
import qualified Data.Vector.Storable.Internal as V

data Statement a where
  Alloc    :: E.DimWrap -> Var a -> Statement a
  Bind     :: Var a -> E.TensorWrap a -> Statement a
  Store    :: Var a -> E.TensorWrap a -> Statement a
  Copy     :: Var a -> Var a -> Statement a
  BlasGEMV :: Order -> Transpose -> VarWithDim D2 a -> VarWithDim D1 a -> a -> a -> VarWithDim D1 a -> Statement a
  BlasGERU :: Order -> VarWithDim D1 a -> VarWithDim D1 a -> a -> VarWithDim D2 a -> Statement a
  BlasGEMM :: Order -> Transpose -> VarWithDim D2 a -> Transpose -> VarWithDim D2 a -> a -> a -> VarWithDim D2 a -> Statement a
  DotAdd   :: Var a -> Var a -> Var a -> Statement a
  DotMul   :: Var a -> Var a -> Var a -> Statement a
  DotSca   :: a -> Var a -> Var a -> Statement a

type VarWithDim d a = (d, Var a)

_vid_VarWithDim :: VarWithDim d a -> Int
_vid_VarWithDim = _vid . snd

deriving instance Data Order
deriving instance Data Transpose
deriving instance Data a => Data (Statement a)

isAlloc  a = constrIndex (toConstr a) == 1
isBind   a = constrIndex (toConstr a) == 2
isStore  a = constrIndex (toConstr a) == 3
isCopy   a = constrIndex (toConstr a) == 4
isGEMV   a = constrIndex (toConstr a) == 5
isGERU   a = constrIndex (toConstr a) == 6
isGEMM   a = constrIndex (toConstr a) == 7
isDotAdd a = constrIndex (toConstr a) == 8
isDotMul a = constrIndex (toConstr a) == 9
isDotSca a = constrIndex (toConstr a) == 10

isBindToTensor t1 s | Bind _ t2 <- s, t1 == t2 = True
                    | otherwise = False

isStoreTo vid s | Store v _ <-s, vid == _vid v = True
                | otherwise = False

isDotAddTo vid s | DotAdd v1 v2 _ <- s, vid == _vid v1 || vid == _vid v2 = True
                 | otherwise = False

isDotScaTo vid s | DotSca a v1 v2 <- s, vid == _vid v1 = True
                 | otherwise = False

substitute :: Element a => Var a -> Var a -> [Statement a] -> [Statement a]
substitute v1 v2 st = if v1 == v2 then st else map subst st
  where
    subst (Store v3 t)
      | _vid v1 == _vid v3 = Store (v3{_vid = _vid v2}) t
    subst (BlasGEMV o t (da,va) (db,vb) a b (dc,vc))
      | _vid v1 == _vid va = BlasGEMV o t (da,va{_vid = _vid v2}) (db,vb) a b (dc,vc)
      | _vid v1 == _vid vb = BlasGEMV o t (da,va) (db,vb{_vid = _vid v2}) a b (dc,vc)
      | _vid v1 == _vid vc = BlasGEMV o t (da,va) (db,vb) a b (dc,vc{_vid = _vid v2})
    -- TODO
    subst (DotAdd va vb vc)
      | _vid v1 == _vid va = DotAdd (va{_vid = _vid v2}) vb vc
      | _vid v1 == _vid vb = DotAdd va (vb{_vid = _vid v2}) vc
      | _vid v1 == _vid vc = DotAdd va vb (vc{_vid = _vid v2})
    subst (DotMul va vb vc)
      | _vid v1 == _vid va = DotMul (va{_vid = _vid v2}) vb vc
      | _vid v1 == _vid vb = DotMul va (vb{_vid = _vid v2}) vc
      | _vid v1 == _vid vc = DotMul va vb (vc{_vid = _vid v2})
    subst (DotSca f va vb)
      | _vid v1 == _vid va = DotSca f (va{_vid = _vid v2}) vb
      | _vid v1 == _vid vb = DotSca f va (vb{_vid = _vid v2})

instance Show (Var a) where
  show (Var i) = printf "v%03d" i

instance Show a => Pretty (Statement a) where
  pretty (Alloc d v) = fill 8 (text "Alloc") <+> text (printf "%s" (show v))
  pretty (Bind  v t) = fill 8 (text "Bind")  <+> text (printf "%s %s" (show v) (show t))
  pretty (Store v t) = fill 8 (text "Store") <+> text (printf "%s %s" (show v) (show t))
  pretty (Copy v1 v2)= fill 8 (text "Copy")  <+> text (printf "%s %s" (show v1) (show v2))
  pretty (BlasGEMV o t v1 v2 a b v3)     = fill 8 (text "Gemv") <+> hsep (map text [show v1, show v2, show v3, show a, show b])
  pretty (BlasGERU o v1 v2 a v3)         = fill 8 (text "Geru") <+> hsep (map text [show v1, show v2, show v3, show a])
  pretty (BlasGEMM o t1 v1 t2 v2 a b v3) = fill 8 (text "Gemm") <+> hsep (map text [show v1, show v2, show v3, show a, show b])
  pretty (DotAdd v1 v2 v3)               = fill 8 (text "Add")  <+> hsep (map text [show v1, show v2, show v3])
  pretty (DotMul v1 v2 v3)               = fill 8 (text "Mul")  <+> hsep (map text [show v1, show v2, show v3])
  pretty (DotSca a  v1 v2)               = fill 8 (text "Sca")  <+> hsep (map text [show a , show v1, show v2])

instance Show a => Show (Statement a) where
  show = show . pretty

type EvalS a = M.Map Int (E.TensorWrap a)
type EvalM a = ExceptT EvalE (StateT (EvalS a) IO)
data EvalE = Fail String
  deriving (Show)

newtype Var a = Var {_vid  :: VarId}
  deriving (Typeable, Data, Eq)
data VarAny where
  VarAny :: Element a => Var a -> VarAny
type CG = StateT CGState (ExceptT CGError IO)
data CGError = CGSizeMismatchedTensors
             | CGMismatchedDimension String String String
             | CGMismatchedElementType
             | CGReferenceUnknown 
             | CGReferenceAlreadyBound
  deriving (Eq, Show)
data CGState = CGS { _cgs_let :: M.Map E.Var VarAny
                   , _cgs_var_counter :: VarId }

runCG :: CGState -> CG a -> IO (Either CGError (a, CGState))
runCG cg act = runExceptT (runStateT act cg)

initCG :: CGState
initCG = CGS M.empty 0

new_var :: CG (Var a)
new_var = do
  i <- _cgs_var_counter <$> get
  modify (\s -> s{ _cgs_var_counter = i + 1 })
  return $ Var i

add_let :: E.Var -> VarAny -> CG ()
add_let ev v = do
  bnds <- _cgs_let <$> get
  if M.member ev bnds
    then
      throwError CGReferenceAlreadyBound
    else
      modify (\s -> s{ _cgs_let = M.insert ev v bnds})

lookup_let :: E.Var -> CG (Maybe VarAny)
lookup_let ev = do
  bnds <- _cgs_let <$> get
  return $ M.lookup ev bnds

toStatements :: Element a => E.ExprHashed a -> CG ([Statement a], Var a)
toStatements (_ E.:@ E.L v e1 e2) = do
  (s1, v1) <- toStatements e1
  add_let v (VarAny v1)
  (s2, v2) <- toStatements e2
  return (s1 ++ [] ++ s2, v2)

toStatements (_ E.:@ E.V v) = do
  mb <- lookup_let v
  case mb of 
    Nothing -> throwError CGReferenceUnknown
    Just (VarAny v) -> 
      case cast v of
        Just v -> return ([], v)
        _      -> throwError CGMismatchedElementType

toStatements (_ E.:@ E.I t) = do
  v <- new_var
  return ([Bind v t], v)

toStatements ((d, _) E.:@ E.S f a) = do
  (s1, v1) <- toStatements a
  v2 <- new_var
  return (s1 ++ [Alloc d v2, DotSca f v1 v2], v2)

toStatements ((d, _) E.:@ E.Bin op e1 e2) = do
  (s1, v1) <- toStatements e1
  (s2, v2) <- toStatements e2
  v3 <- new_var
  let dw1 = E.attrDim e1
      dw2 = E.attrDim e2
  instr <- case op of
     E.DM  -> return $ DotMul v1 v2 v3
     E.DA  -> return $ DotAdd v1 v2 v3
     E.VM  | E.DimWrap d1 <- dw1, Just (d1 :: D1) <- cast d1
           , E.DimWrap d2 <- dw2, Just (d2 :: D2) <- cast d2 
           , E.DimWrap d3 <- d  , Just (d3 :: D1) <- cast d3 -> return $ BlasGEMV RowMajor Trans (d2, v2) (d1, v1) 1 0 (d3, v3)
     E.MV  | E.DimWrap d1 <- dw1, Just d1 <- cast d1
           , E.DimWrap d2 <- dw2, Just d2 <- cast d2 
           , E.DimWrap d3 <- d  , Just d3 <- cast d3 -> return $ BlasGEMV RowMajor NoTrans (d1, v1) (d2, v2) 1 0 (d3, v3)
     E.MTM | E.DimWrap d1 <- dw1, Just d1 <- cast d1
           , E.DimWrap d2 <- dw2, Just d2 <- cast d2 
           , E.DimWrap d3 <- d  , Just d3 <- cast d3 -> return $ BlasGEMM ColMajor Trans (d1, v1) NoTrans (d2, v2) 1 0 (d3, v3)
     E.OVV | E.DimWrap d1 <- dw1, Just d1 <- cast d1
           , E.DimWrap d2 <- dw2, Just d2 <- cast d2 
           , E.DimWrap d3 <- d  , Just d3 <- cast d3 -> return $ BlasGERU RowMajor (d1, v1) (d2, v2) 1 (d3, v3)
     _ -> throwError (CGMismatchedDimension (show dw1) (show dw2) (show d))
  return (s1 ++ s2 ++ [Alloc d v3, instr], v3)

execute :: Element a => [Statement a] -> IO (Either EvalE ())
execute ss = evalStateT (runExceptT $ eval ss) M.empty
  where
    eval [] = return ()
    eval (s:ss) = do
      case s of
        Alloc (E.DimWrap (d :: d)) (v :: Var a) -> do
          m <- get
          if M.member (_vid v) m
            then throwError $ Fail (show $ pretty s)
            else do
              (t :: Tensor d a) <- liftIO $ newTensor d
              put $ M.insert (_vid v) (E.TensorWrap t) m
        Bind  v (E.TensorWrap t) -> do
          m <- get
          if M.member (_vid v) m
            then throwError $ Fail (show $ pretty s)
            else put $ M.insert (_vid v) (E.TensorWrap t) m
        Store v t -> do
          m <- get
          case M.lookup (_vid v) m of
            Just ts | E.TensorWrap ts <- ts
                    , E.TensorWrap td <- t
                    , size (_tdim ts) == size (_tdim td)
                    , Just td <- cast td
                    -> liftIO $ copyTensor ts td
            _       -> throwError $ Fail (show $ pretty s)
        Copy  v w -> do
          m <- get
          let vt = M.lookup (_vid v) m
              wt = M.lookup (_vid w) m
          case () of
            _ | Just (E.TensorWrap vt) <- vt
              , Just (E.TensorWrap wt) <- wt
              , size (_tdim vt) == size (_tdim wt)
              , Just wt <- cast wt
              -> liftIO $ copyTensor vt wt
            _ -> throwError $ Fail (show $ pretty s)
        BlasGEMV o t (dv,v) (dw,w) a b (du,u) ->
          dotop2 (throwError $ Fail (show $ pretty s)) v w u (\tx ty tz -> liftIO $
            V.unsafeWith (_tdat tx) (\px ->
            V.unsafeWith (_tdat ty) (\py ->
            V.unsafeWith (_tdat tz) (\pz ->
              let (D2 r c) = dv
                  (m,n,lda) = case o of
                                RowMajor -> (r,c,c)
                                ColMajor -> (c,r,c)
              in gemv o t m n a px lda py 1 b pz 1))))
        BlasGERU o (dv,v) (dw,w) a (du,u) ->
          dotop2 (throwError $ Fail (show $ pretty s)) v w u (\tx ty tz -> liftIO $
            V.unsafeWith (_tdat tx) (\px ->
            V.unsafeWith (_tdat ty) (\py ->
            V.unsafeWith (_tdat tz) (\pz ->
              let D1 r = dv
                  D1 c = dw
                  (m,n,lda) = case o of
                                RowMajor -> (r, c, c)
                                ColMajor -> (c, r, c)
              in geru o m n a px 1 py 1 pz lda))))
        BlasGEMM o t1 (d1,v1) t2 (d2,v2) a b (dw,w) ->
          dotop2 (throwError $ Fail (show $ pretty s)) v1 v2 w (\tx ty tz -> liftIO $
            V.unsafeWith (_tdat tx) (\px ->
            V.unsafeWith (_tdat ty) (\py ->
            V.unsafeWith (_tdat tz) (\pz ->
              let D2 r1 c1 = d1
                  D2 r2 c2 = d2
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
        DotAdd v w u -> dotop2 (throwError $ Fail (show $ pretty s)) v w u (\tv tw tu -> liftIO $
                          hadamard plus  (_tdat tu) (_tdat tv) (_tdat tw))
        DotMul v w u -> dotop2 (throwError $ Fail (show $ pretty s)) v w u (\tv tw tu -> liftIO $
                          hadamard times (_tdat tu) (_tdat tv) (_tdat tw))
        DotSca a v w -> dotop1 (throwError $ Fail (show $ pretty s)) v w (\tv tw -> liftIO $
                          foreach (times (konst a)) (_tdat tw) (_tdat tv))
      eval ss

    dotop1 :: EvalM a b 
           -> Var a 
           -> Var a 
           -> (forall d1 d2. (Dimension d1, Dimension d2) => Tensor d1 a -> Tensor d2 a -> EvalM a b) 
           -> EvalM a b
    dotop1 e v w o = do
      m <- get
      let vt = M.lookup (_vid v) m
          wt = M.lookup (_vid w) m
      case (vt, wt) of
        _ | Just (E.TensorWrap vt) <- vt
          , Just (E.TensorWrap wt) <- wt
          -> o vt wt
        _ -> e

    dotop2 :: EvalM a b 
           -> Var a 
           -> Var a 
           -> Var a 
           -> (forall d1 d2 d3. (Dimension d1, Dimension d2, Dimension d3) => Tensor d1 a -> Tensor d2 a -> Tensor d3 a -> EvalM a b) 
           -> EvalM a b
    dotop2 e u v w o = do
      m <- get
      let ut = M.lookup (_vid u) m
          vt = M.lookup (_vid v) m
          wt = M.lookup (_vid w) m
      case (ut, vt, wt) of
        _ | Just (E.TensorWrap ut) <- ut
          , Just (E.TensorWrap vt) <- vt
          , Just (E.TensorWrap wt) <- wt
          -> o ut vt wt
        _ -> e
