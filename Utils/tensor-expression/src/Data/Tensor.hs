module Data.Tensor(
  Dimension(..), Element(..), Tensor(..),
  D1(..), D2(..), D3(..),
  newTensor, packTensor, copyTensor, eqTensor,
  Expr(..), dim, Statement(..),
  toExprHashed, elimCommonExpr, qualify, toStatements, optimize, CGState, runCG, initCG, execute,
  compile, eval, eval',
) where

import Data.Tensor.Class
import Data.Tensor.Compile hiding (Var)
import Data.Tensor.Optimize
import Data.Tensor.Execute
import Data.Either
import Control.Monad.State (runStateT)
import Control.Monad.Trans (liftIO)

compile :: (Dimension d, Element a) => Expr d a -> CG ([Statement a], Var a)
compile = toStatements . uncurry qualify . elimCommonExpr . toExprHashed

compileAndSave :: (Dimension d, Element a) => Expr d a -> Tensor d a -> CG ([Statement a])
compileAndSave e t = do 
  (st, v) <- compile e
  return (st ++ [Store v (TensorWrap t)])


eval :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
eval expr = do
  t <- newTensor (dim expr)
  handleE $ runCG initCG $ do
    st <- compileAndSave expr t
    st <- optimize st
    liftIO $ execute st
  return t

eval' :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
eval' expr = do
  t <- newTensor (dim expr)
  handleE $ runCG initCG $ do
     st <- compileAndSave expr t
     liftIO $ execute st
  return t

handleE :: Show e => IO (Either e a) -> IO a
handleE act = act >>= either (ioError . userError . show) return
