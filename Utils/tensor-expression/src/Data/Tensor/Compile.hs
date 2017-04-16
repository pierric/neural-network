module Data.Tensor.Compile where

import Data.Hashable
import Text.Printf
import Control.Monad.State.Strict
import Control.Monad.Writer.Strict
import qualified Data.Vector.Storable.Mutable  as V
import Data.Typeable (cast)
import Foreign.ForeignPtr (castForeignPtr)
import System.IO.Unsafe (unsafeDupablePerformIO)
import Text.PrettyPrint.Free (Pretty(..), Doc, hang, text, above, (<+>))
import qualified Data.Tensor.Class as U
import Data.Tensor.Class hiding (Expr(..))

newtype TensorWrap e = TensorWrap (V.IOVector e)

instance V.Storable e => Hashable (TensorWrap e) where
  hashWithSalt s (TensorWrap v) = s `hashWithSalt` V.length v `hashWithSalt` (unsafeDupablePerformIO $ V.unsafeWith v return)

instance V.Storable e => Eq (TensorWrap e) where
  TensorWrap t1@(V.MVector o1 p1) == TensorWrap t2@(V.MVector o2 p2) =
    V.length t1 == V.length t2 && o1 == o2 && p1 == castForeignPtr p2

instance V.Storable e => Show (TensorWrap e) where
  show (TensorWrap t@(V.MVector o v)) = printf "<tensor_internal (%8d): %s + %4d>" (V.length t) (show v) o

infix 7 :@
infix 7 :>
infix 3 :%

data ExprAttr a e = a :@ ExprBody e (ExprAttr a e)
  deriving (Eq)
data ExprBody e x = L VarId x x
                  | V VarId
                  | I (TensorWrap e)
                  | S e x
                  | Bin ExprOp x x
  deriving (Eq)
data ExprOp = DM | DA | VM | MV | MTM | OVV
  deriving (Enum, Eq)

attr :: ExprAttr a e -> a
attr (a :@ _) = a

body :: ExprAttr a e -> ExprBody e (ExprAttr a e)
body (_ :@ b) = b

-- Cxt is the derivative of ExprAttr
data Cxt  a e = Nil | Cxt a e :> Path a e (ExprAttr a e)
data Path a e x = PLL a VarId x
                | PLR a VarId x
                | PS  a e
                | PBL a ExprOp x
                | PBR a ExprOp x
-- The Cxt and the (sub-)tree at the focus
data ExprWithCxt a e = Cxt a e :% ExprAttr a e

root :: ExprWithCxt a e -> ExprWithCxt a e
root = go
  where
    go (cxt :> PLL a v x :% e) = go $ cxt :% a :@ L v e x
    go (cxt :> PLR a v x :% e) = go $ cxt :% a :@ L v x e
    go (cxt :> PS  a f   :% e) = go $ cxt :% a :@ S f e
    go (cxt :> PBL a o x :% e) = go $ cxt :% a :@ Bin o e x
    go (cxt :> PBR a o x :% e) = go $ cxt :% a :@ Bin o x e
    go (Nil :% e) = Nil :% e

head_po :: ExprWithCxt a e -> ExprWithCxt a e
head_po = go
  where
    go (cxt :% a :@ L v x y)   = go $ cxt :> PLL a v y :% x
    go (cxt :% a :@ S e x  )   = go $ cxt :> PS  a e   :% x
    go (cxt :% a :@ Bin o x y) = go $ cxt :> PBL a o y :% x
    go ec = ec

next_po :: ExprWithCxt a e -> Maybe (ExprWithCxt a e)
next_po (Nil :% _) = Nothing
next_po (cxt :> PLL a v x :% e) = Just $ head_po $ cxt :> PLR a v e :% x
next_po (cxt :> PLR a v x :% e) = Just $ cxt :% a :@ L v x e
next_po (cxt :> PS  a f   :% e) = Just $ cxt :% a :@ S f e
next_po (cxt :> PBL a o x :% e) = Just $ head_po $ cxt :> PBR a o e :% x
next_po (cxt :> PBR a o x :% e) = Just $ cxt :% a :@ Bin o x e

type CM = StateT VarId IO

runCM :: CM a -> IO a
runCM = flip evalStateT 0

newVar :: CM VarId
newVar = do
  i <- get
  modify (+1)
  return $ i

eliminate_common_expr :: Element e => ExprHashed e -> CM (ExprHashed e)
eliminate_common_expr e = do
  Nil :% e <- rp . head_po $ Nil :% e
  return e
  where
    rp ec = do
      ec <- step ec
      maybe (return $ root ec) rp (next_po ec)

    -- at a position of the expression, try matching with the tree at focus
    -- to all following positions (post-order), resulting in possibly updated
    -- the same position.
    --
    -- Note that it is not necessary to match I and V, for they cost nothing.
    -- And L neither, because it won't appear later.
    step tc@(cxt :% _ :@ V{}) = return tc
    step tc@(cxt :% _ :@ L{}) = return tc
    step tc@(cxt :% _ :@ I{}) = return tc
    step    (cxt :% e  ) = do
      v <- mk (dim_ce e) . V <$> newVar
      (cxt, upd) <- runWriterT $ go cxt (subst v e)
      return (cxt :% if getAny upd then v else e)
      where
        go Nil              _ = return Nil
        go (c :> PLL a v x) f = do c <- go c f
                                   x <- f x
                                   return $ c :> PLL a v x
        go (c :> PLR a v x) f = do c <- go c f
                                   x <- f x
                                   return $ c :> PLR a v x
        go (c :> PS a e)    f = do c <- go c f
                                   return $ c :> PS a e
        go (c :> PBL a v x) f = do c <- go c f
                                   x <- f x
                                   return $ c :> PBL a v x
        go (c :> PBR a v x) f = do c <- go c f
                                   x <- f x
                                   return $ c :> PBR a v x

        subst v a b = if hash_ce a == hash_ce b
                      then tell (Any True) >> return v
                      else case b of
                        u :@ L   w x y -> do x <- subst v a x
                                             y <- subst v a y
                                             return $ u :@ L   w x y
                        u :@ S   w x   -> do x <- subst v a x
                                             return $ u :@ S   w x
                        u :@ Bin w x y -> do x <- subst v a x
                                             y <- subst v a y
                                             return $ u :@ Bin w x y
                        _              -> return b

data DimWrap where
  DimWrap :: Dimension d => d -> DimWrap

instance Hashable DimWrap where
  hashWithSalt s (DimWrap d) = hashWithSalt s d

deriving instance Show DimWrap

instance Eq DimWrap where
  DimWrap d1 == DimWrap d2 = case cast d1 of
                               Nothing -> False
                               Just d1 -> d1 == d2

type ExprHashed = ExprAttr (DimWrap, Int)

dim_ce  = fst. attr
hash_ce = snd . attr

compile :: (Dimension d, Element e) => U.Expr d e -> ExprHashed e
compile e@(U.I x)     = mk (DimWrap $ dim e) $ I (TensorWrap $ _tdat x)
compile e@(U.S x y)   = mk (DimWrap $ dim e) $ S x (compile y)
compile e@(x U.:.* y) = mk (DimWrap $ dim e) $ Bin DM  (compile x) (compile y)
compile e@(x U.:.+ y) = mk (DimWrap $ dim e) $ Bin DA  (compile x) (compile y)
compile e@(x U.:<# y) = mk (DimWrap $ dim e) $ Bin VM  (compile x) (compile y)
compile e@(x U.:#> y) = mk (DimWrap $ dim e) $ Bin MV  (compile x) (compile y)
compile e@(x U.:%# y) = mk (DimWrap $ dim e) $ Bin MTM (compile x) (compile y)
compile e@(x U.:<> y) = mk (DimWrap $ dim e) $ Bin OVV (compile x) (compile y)

mk :: Element e => DimWrap -> ExprBody e (ExprHashed e) -> ExprHashed e
mk d b@(I t)       = (d, iExpr `hashWithSalt` t) :@ b
mk d b@(V v)       = (d, vExpr `hashWithSalt` v) :@ b
mk d b@(L v x y)   = (d, lExpr `hashWithSalt` v `hashWithSalt` attr x `hashWithSalt` attr y) :@ b
mk d b@(S e x)     = (d, sExpr `hashWithSalt` e `hashWithSalt` attr x) :@ b
mk d b@(Bin o x y) = (d, bExpr `hashWithSalt` fromEnum o `hashWithSalt` attr x `hashWithSalt` attr y) :@ b

iExpr, vExpr, lExpr, sExpr, bExpr :: Int
iExpr = 0
vExpr = 1
lExpr = 2
sExpr = 3
bExpr = 4

instance Pretty DimWrap where
  pretty (DimWrap d) = pretty d

instance (Pretty e, V.Storable e) => Pretty (ExprHashed e) where
  pretty (a :@ (L v x y))   = hang 4 $ (pretty (fst a) <+> text "L")
                                `above` pretty x
                                `above` pretty y
  pretty (a :@ (V v))       = hang 4 $ (pretty (fst a) <+> text "V")
  pretty (a :@ (I i))       = hang 4 $ (pretty (fst a) <+> text "I")
                                `above` text (show i)
  pretty (a :@ (S f x))     = hang 4 $ (pretty (fst a) <+> text "S")
                                `above` pretty x
  pretty (a :@ (Bin o x y)) = hang 4 $ (pretty (fst a) <+> text "B" <+> pretty o)
                                `above` pretty x
                                `above` pretty y

instance Pretty ExprOp where
  pretty DM  = text ".*"
  pretty DA  = text ".+"
  pretty VM  = text "<#"
  pretty MV  = text "#>"
  pretty MTM = text "ᵀ×"
  pretty OVV = text "⊗"

instance (Pretty e, V.Storable e) => Show (ExprHashed e) where
  show = show . pretty