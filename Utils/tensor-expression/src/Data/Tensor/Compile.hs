module Data.Tensor.Compile where

import Data.Hashable
import qualified Data.Tensor.Class as C
import Data.Tensor.Class hiding (Expr(..))

data Expr e d a where
  -- additional sytax to allow a lift of common sub-expression
  L :: e -> Var d a -> Expr e d a -> Expr e d a -> Expr e d a
  V :: e -> Var d a -> Expr e d a

  I :: e -> Tensor d a -> Expr e d a
  S :: e -> a -> Expr e d a -> Expr e d a
  DM :: e -> Expr e d a -> Expr e d a -> Expr e d a
  DA :: e -> Expr e d a -> Expr e d a -> Expr e d a
  VM :: e -> Expr e D1 a -> Expr e D2 a -> Expr e D1 a
  MV :: e -> Expr e D2 a -> Expr e D1 a -> Expr e D1 a
  MTM :: e -> Expr e D2 a -> Expr e D2 a -> Expr e D2 a
  OVV :: e -> Expr e D1 a -> Expr e D1 a -> Expr e D2 a

type ExprWithHash =  Expr Int

compile :: (Hashable (C.Expr d a), Dimension d, Element a) => C.Expr d a -> ExprWithHash d a
compile e@(C.I x)   = I   (hash e) x
compile e@(C.S x y) = S   (hash e) x (compile y)
compile e@(x C.:.* y) = DM  (hash e) (compile x) (compile y)
compile e@(x C.:.+ y) = DA  (hash e) (compile x) (compile y)
compile e@(x C.:<# y) = VM  (hash e) (compile x) (compile y)
compile e@(x C.:#> y) = MV  (hash e) (compile x) (compile y)
compile e@(x C.:%# y) = MTM (hash e) (compile x) (compile y)
compile e@(x C.:<> y) = OVV (hash e) (compile x) (compile y)
