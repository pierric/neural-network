module Data.Tensor(
  Dimension(..), Element(..), Tensor(..),
  D1(..), D2(..), D3(..),
  newTensor, packTensor, copyTensor, eqTensor,
  Expr(..), dim, Statement(..),
  toExprHashed, elimCommonExpr, close, toStatements, 
  CGState, runCG, initCG, 
  compile, compileAndStore, optimize, execute, 
  eval, evalNoOpt,
) where

import Data.Tensor.Class
import Data.Tensor.Compile hiding (Var)
import Data.Tensor.Optimize
import Data.Tensor.Execute
import Data.Either
import Control.Monad.State (runStateT)
import Control.Monad.Trans (liftIO)

compile :: (Dimension d, Element a) => Expr d a -> CG ([Statement a], Var a)
compile = toStatements . close . elimCommonExpr . toExprHashed

compileAndStore :: (Dimension d, Element a) => Expr d a -> Tensor d a -> CG ([Statement a])
compileAndStore e t = do 
  (st, v) <- compile e
  return (st ++ [Store v (TensorWrap t)])


eval :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
eval expr = do
  t <- newTensor (dim expr)
  handleE $ runCG initCG $ do
    st <- compileAndStore expr t
    st <- optimize st
    liftIO $ execute st
  return t

evalNoOpt :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
evalNoOpt expr = do
  t <- newTensor (dim expr)
  handleE $ runCG initCG $ do
     st <- compileAndStore expr t
     liftIO $ execute st
  return t

handleE :: Show e => IO (Either e a) -> IO a
handleE act = act >>= either (ioError . userError . show) return
