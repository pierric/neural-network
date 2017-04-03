module Data.Tensor(
  Dimension(..), Element(..), Tensor(..),
  D1(..), D2(..), D3(..),
  newTensor, copyTensor, tensor_eq,
  Expr(..), Statement,
  compile, optimize, CGState,
) where

import Data.Tensor.Class
import Data.Tensor.Compile
import Data.Tensor.Optimize

execute :: (Dimension d, Element a) => Expr d a -> IO (Tensor d a)
execute expr = undefined
