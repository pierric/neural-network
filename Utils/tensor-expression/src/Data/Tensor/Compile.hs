module Data.Tensor.Compile where

import Data.Hashable
import Control.Monad.State.Strict
import qualified Data.Tensor.Class as C
import Data.Tensor.Class hiding (Expr(..))

data TensorWrap e where
  TensorWrap :: (Dimension d, Element e) => Tensor d e -> TensorWrap e

instance Hashable e => Hashable (TensorWrap e) where
  hashWithSalt s (TensorWrap t) = s `hashWithSalt` t


infix 7 :@
infix 7 :>
infix 3 :%

data ExprAttr a e = a :@ ExprBody e (ExprAttr a e)
data ExprBody e x = L VarId x x
                  | V VarId
                  | I (TensorWrap e)
                  | S e x
                  | Bin ExprOp x x
data ExprOp = DM | DA | VM | MV | MTM | OVV
  deriving Enum

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
      v <- mk . V <$> newVar
      let cxt' = go cxt (subst v e)
      return (cxt' :% v)
      where
        go Nil              _ = Nil
        go (c :> PLL a v x) f = go c f :> PLL a v (f x)
        go (c :> PLR a v x) f = go c f :> PLR a v x
        go (c :> PS a e)    f = go c f :> PS a e
        go (c :> PBL a v x) f = go c f :> PBL a v (f x)
        go (c :> PBR a v x) f = go c f :> PBR a v x

        subst v a b = if attr a == attr b
                      then v
                      else case b of
                        u :@ L   w x y -> u :@ L   w (subst v a x) (subst v a y)
                        u :@ S   w x   -> u :@ S   w (subst v a x)
                        u :@ Bin w x y -> u :@ Bin w (subst v a x) (subst v a y)

type ExprHashed = ExprAttr Int
compile :: (Dimension d, Element e) => C.Expr d e -> ExprHashed e
compile (C.I x)     = mk $ I (TensorWrap x)
compile (C.S x y)   = mk $ S x (compile y)
compile (x C.:.* y) = mk $ Bin DM  (compile x) (compile y)
compile (x C.:.+ y) = mk $ Bin DA  (compile x) (compile y)
compile (x C.:<# y) = mk $ Bin VM  (compile x) (compile y)
compile (x C.:#> y) = mk $ Bin MV  (compile x) (compile y)
compile (x C.:%# y) = mk $ Bin MTM (compile x) (compile y)
compile (x C.:<> y) = mk $ Bin OVV (compile x) (compile y)

mk :: Element e => ExprBody e (ExprHashed e) -> ExprHashed e
mk b@(I t)       = (iExpr `hashWithSalt` t) :@ b
mk b@(V v)       = (vExpr `hashWithSalt` v) :@ b
mk b@(L v x y)   = (lExpr `hashWithSalt` v `hashWithSalt` attr x `hashWithSalt` attr y) :@ b
mk b@(S e x)     = (sExpr `hashWithSalt` e `hashWithSalt` attr x) :@ b
mk b@(Bin o x y) = (bExpr `hashWithSalt` fromEnum o `hashWithSalt` attr x `hashWithSalt` attr y) :@ b

iExpr, vExpr, lExpr, sExpr, bExpr :: Int
iExpr = 0
vExpr = 1
lExpr = 2
sExpr = 3
bExpr = 4
