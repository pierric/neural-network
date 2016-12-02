{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
module Data.NeuronNetwork.Backend.BLASHS (
  module Data.NeuronNetwork.Backend.BLASHS.Layers,
  ByBLASHS(..),
  ErrCode(..)
) where

import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.BLASHS.Layers
import Control.Monad.Except

data ErrCode = ErrMismatch
type Err     = ExceptT ErrCode IO

-- the backend type
data ByBLASHS = ByBLASHS

-- with 1D input
instance (TranslateBody s, Component (RunLayer (SpecToTag s))) =>
    Backend ByBLASHS (SpecIn1D :++ s) where
  type Env ByBLASHS = Err
  type ConvertFromSpec (SpecIn1D :++ s) = RunLayer (SpecToTag s)
  compile _ (a :++ l)= trans (size Nothing a) l

-- with 2D input
instance (TranslateBody s, Component (RunLayer (SpecToTag s))) =>
    Backend ByBLASHS (SpecIn2D :++ s) where
  type Env ByBLASHS = Err
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

-- translate the body of specification
class TranslateBody s where
  type SpecToTag s
  trans :: Size -> s -> Err (RunLayer (SpecToTag s))

instance TranslateBody SpecFullConnect where
  type SpecToTag SpecFullConnect = S F (T (SinglC :. DenseVector))
  trans (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                    return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecReshape2DAs1D where
  type SpecToTag SpecReshape2DAs1D = A
  trans (D2 _ _ _) _ = return As1D
  trans (D1 _)     _ = throwError ErrMismatch

instance (TranslateBody a, TranslateBody c, ComputeSize a) => TranslateBody (a :++ c) where
  type SpecToTag (a :++ b) = S (SpecToTag a) (SpecToTag b)
  trans s (a :++ c) = do u <- trans s a
                         v <- trans (size (Just s) a) c
                         return $ Stack u v
