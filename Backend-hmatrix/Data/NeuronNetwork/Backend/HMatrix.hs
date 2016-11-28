-- {-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
module Data.NeuronNetwork.Backend.HMatrix (
  module Data.NeuronNetwork.Backend.HMatrix.Layers,
  ByHmatrix(..),
  Err(..),
  M
) where

import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.HMatrix.Utils
import Data.NeuronNetwork.Backend.HMatrix.Layers

import Control.Monad.Except

data Err = ErrMismatch
type M   = ExceptT Err IO

-- the backend type
data ByHmatrix = ByHmatrix

-- with 1D input
instance (TranslateBody s, Component (RunLayer (SpecToTag s))) =>
    Backend ByHmatrix (SpecIn1D :++ s) where
  type Env ByHmatrix = M
  type ConvertFromSpec (SpecIn1D :++ s) = RunLayer (SpecToTag s)
  compile _ (a :++ l)= trans (size Nothing a) l

-- with 2D input
instance (TranslateBody s, Component (RunLayer (SpecToTag s))) =>
    Backend ByHmatrix (SpecIn2D :++ s) where
  type Env ByHmatrix = M
  type ConvertFromSpec (SpecIn2D :++ s) = RunLayer (SpecToTag s)
  compile _ (a :++ l)= trans (size Nothing a) l

-- It is necessary to propagate the size along the layers,
-- because fullconnect and convolution need to know
-- the previous size.
data Size = D1 Int | D2 Int Int Int

class ComputeSize l where
  size :: Maybe Size -> l -> Size
instance ComputeSize SpecIn1D where
  size Nothing (In1D n) = D1 n
instance ComputeSize SpecIn2D where
  size Nothing (In2D m n) = D2 1 m n
instance ComputeSize SpecReshape2DAs1D where
  size (Just (D2 k m n)) _ = D1 (k*m*n)
instance ComputeSize SpecFullConnect where
  size _ (FullConnect n)   = D1 n
instance ComputeSize SpecConvolution where
  size (Just (D2 _ m n)) (Convolution k f p) = D2 k (m+2*p-f+1) (n+2*p-f+1)

-- translate the body of specification
class TranslateBody s where
  type SpecToTag s
  trans :: Size -> s -> M (RunLayer (SpecToTag s))

instance TranslateBody SpecFullConnect where
  type SpecToTag SpecFullConnect = F
  trans (D1 s) (FullConnect n) = lift $ newDLayer (s,n) (relu, relu')
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecConvolution where
  type SpecToTag SpecConvolution = C
  trans (D2 k s t) (Convolution n f p) = lift $ newCLayer k n f p
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecReshape2DAs1D where
  type SpecToTag SpecReshape2DAs1D  = A
  trans (D2 _ _ _) _ = return As1D
  trans (D1 _)     _ = throwError ErrMismatch

instance (TranslateBody a, TranslateBody c, ComputeSize a) => TranslateBody (a :++ c) where
  type SpecToTag (a :++ b) = S (SpecToTag a) (SpecToTag b)
  trans s (a :++ c) = do u <- trans s a
                         v <- trans (size (Just s) a) c
                         return $ Stack u v
