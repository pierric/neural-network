{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
module Data.NeuralNetwork.Backend.BLASHS (
  module Data.NeuralNetwork.Backend.BLASHS.Layers,
  module Data.NeuralNetwork.Backend.BLASHS.Utils,
  ByBLASHS(..),
  ErrCode(..),
  cost'
) where

import Data.NeuralNetwork hiding (relu, relu', cost')
import Data.NeuralNetwork.Backend.BLASHS.Layers
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD
import Control.Monad.Except
import Data.Constraint (Dict(..))

data ErrCode = ErrMismatch
type Err     = ExceptT ErrCode IO

-- the backend type
data ByBLASHS = ByBLASHS

-- with 1D input
instance (TranslateBody s, Component (RunLayer (SpecToTag s)), Run (RunLayer (SpecToTag s)) ~ IO) =>
    Backend ByBLASHS (SpecIn1D :++ s) where
  type Env ByBLASHS = Err
  type ConvertFromSpec (SpecIn1D :++ s) = RunLayer (SpecToTag s)
  compile _ (a :++ l)= trans (lsize Nothing a) l
  witness _ _ = Dict

-- with 2D input
instance (TranslateBody s, Component (RunLayer (SpecToTag s)), Run (RunLayer (SpecToTag s)) ~ IO) =>
    Backend ByBLASHS (SpecIn2D :++ s) where
  type Env ByBLASHS = Err
  type ConvertFromSpec (SpecIn2D :++ s) = RunLayer (SpecToTag s)
  compile _ (a :++ l)= trans (lsize Nothing a) l
  witness _ _ = Dict

instance RunInEnv IO Err where
  run = liftIO

-- It is necessary to propagate the size along the layers,
-- because fullconnect and convolution need to know
-- the previous size.
data LayerSize = D1 Int | D2 Int Int Int

class ComputeSize l where
  lsize :: Maybe LayerSize -> l -> LayerSize
instance ComputeSize SpecIn1D where
  lsize Nothing (In1D n) = D1 n
instance ComputeSize SpecIn2D where
  lsize Nothing (In2D m n) = D2 1 m n
instance ComputeSize SpecReshape2DAs1D where
  lsize (Just (D2 k m n)) _ = D1 (k*m*n)
instance ComputeSize SpecFullConnect where
  lsize _ (FullConnect n)   = D1 n
instance ComputeSize SpecConvolution where
  lsize (Just (D2 _ m n)) (Convolution k f p) = D2 k (m+2*p-f+1) (n+2*p-f+1)
instance ComputeSize SpecMaxPooling where
  lsize (Just (D2 k m n)) (MaxPooling s) = D2 k (m `div` s) (n `div` s)

-- translate the body of specification
class TranslateBody s where
  type SpecToTag s
  trans :: LayerSize -> s -> Err (RunLayer (SpecToTag s))

instance TranslateBody SpecFullConnect where
  type SpecToTag SpecFullConnect = S F (T SinglVec)
  trans (D1 s) (FullConnect n) = do u <- lift $ newFLayer s n
                                    return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecConvolution where
  type SpecToTag SpecConvolution = S C (T MultiMat)
  trans (D2 k s t) (Convolution n f p) = do u <- lift $ newCLayer k n f p
                                            return $ Stack u (Activation (relu, relu'))
  trans _ _ = throwError ErrMismatch

instance TranslateBody SpecMaxPooling where
  type SpecToTag SpecMaxPooling = P
  trans (D2 _ _ _) (MaxPooling n) = return (MaxP n)
  trans (D1 _)     _              = throwError ErrMismatch

instance TranslateBody SpecReshape2DAs1D where
  type SpecToTag SpecReshape2DAs1D = A
  trans (D2 _ _ _) _ = return As1D
  trans (D1 _)     _ = throwError ErrMismatch

instance (TranslateBody a, TranslateBody c, ComputeSize a) => TranslateBody (a :++ c) where
  type SpecToTag (a :++ b) = S (SpecToTag a) (SpecToTag b)
  trans s (a :++ c) = do u <- trans s a
                         v <- trans (lsize (Just s) a) c
                         return $ Stack u v
