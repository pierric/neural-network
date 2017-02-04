{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
module Data.NeuralNetwork.Backend.HMatrix (
  module Data.NeuralNetwork.Backend.HMatrix.Layers,
  ByHmatrix(..),
  ErrCode(..)
) where

import Data.NeuralNetwork
import Data.NeuralNetwork.Common
import Data.NeuralNetwork.Backend.HMatrix.Layers
import Numeric.LinearAlgebra (Vector, Matrix)
import Control.Monad.Except
import Data.Functor.Identity
import Data.Constraint (Dict(..))

type Err     = ExceptT ErrCode IO

-- the backend type
data ByHmatrix = ByHmatrix

instance (HeadSize z, TranslateBody Err s,
          Component (SpecToCom s),
          RunInEnv (Run (SpecToCom s)) Err)
       => Backend ByHmatrix (z :++ s) where
  type Env ByHmatrix = Err
  type ConvertFromSpec ByHmatrix (z :++ s) = SpecToCom s
  compile _ (a :++ l)= trans (hsize a) l
  witness _ _ = Dict

instance RunInEnv Identity Err where
  run = return . runIdentity

instance TranslateBody Err SpecFullConnect where
  type SpecToCom SpecFullConnect = RunLayer (S F (T (SinglC :. Vector)))
  trans (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                    return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody Err SpecConvolution where
  type SpecToCom SpecConvolution = RunLayer (S C (T (MultiC :. Matrix)))
  trans (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p
                                            return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody Err SpecReshape2DAs1D where
  type SpecToCom SpecReshape2DAs1D = RunLayer A
  trans (D2 _ _ _) _ = return As1D
  trans (D1 _)     _ = throwError ErrMismatch

instance TranslateBody Err SpecMaxPooling where
  type SpecToCom SpecMaxPooling = RunLayer M
  trans (D2 _ _ _) (MaxPooling n) = return (MaxP n)
  trans (D1 _)     _              = throwError ErrMismatch
