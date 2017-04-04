module Data.Tensor(
  Dimension(..), Element(..), Tensor(..),
  D1(..), D2(..), D3(..),
  newTensor, packTensor, copyTensor, eqTensor,
  Expr(..), Statement,
  compile, optimize, CGState,
  execute, execute',
) where

import Data.Tensor.Class
import Data.Tensor.Compile
import Data.Tensor.Optimize
import Data.Either
import Control.Monad.State (runStateT)

execute :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
execute expr = do
  t <- newTensor (dim expr)
  ((st, v), cg) <- handleE $ runCG 0 (compile expr)
  st' <- optimize cg (st ++ [Store v t])
  handleE $ evaluate st'
  return t

execute' :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
execute' expr = do
  t <- newTensor (dim expr)
  ((st, v), cg) <- handleE $ runCG 0 (compile expr)
  let st' = st ++ [Store v t]
  handleE $ evaluate st'
  return t

handleE :: Show e => IO (Either e a) -> IO a
handleE act = act >>= either (ioError . userError . show) return
