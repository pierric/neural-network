------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Stack
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module defines an general mechanism to stack two
-- compatible neural network component.
------------------------------------------------------------
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module Data.NeuralNetwork.Stack (
  Stack(..),
) where

import Data.HVect
import GHC.TypeLits
import Control.Monad.Trans
import Control.Monad.Except (MonadError)
import Data.NeuralNetwork.Model

data Stack a b = Stack a b

instance (Component a, Component b,
          Monad (Run a), Monad (Run b),
          Dty a ~ Dty b,
          Run a ~ Run b,
          Out a ~ Inp b
         ) => Component (Stack a b) where
  type Run (Stack a b) = Run a
  type Inp (Stack a b) = Inp a
  type Out (Stack a b) = Out b
  newtype Trace (Stack a b) = S0Trace (Trace b, Trace a)
  forwardT (Stack a b) !i = do
    !tra <- forwardT a i
    !trb <- forwardT b (output tra)
    return $ S0Trace (trb, tra)
  output (S0Trace !a) = output (fst a)
  backward (Stack a b) (S0Trace (!trb,!tra)) !odeltb rate = do
    (b', !odelta) <- backward b trb odeltb rate
    (a', !idelta) <- backward a tra odelta rate
    return (Stack a' b', idelta)


-- internal type class to do induction on a non-empty hvect
class HVectStackable k a b where
  type HVectSpecToCom k a b
  hvectTrans :: k -> LayerSize -> HVect (a ': b) -> Err (HVectSpecToCom k a b)

instance BodyTrans k a => HVectStackable k a '[] where
  type HVectSpecToCom k a '[] = SpecToCom k a
  hvectTrans bk sz (a :&: HNil) = btrans bk sz a

instance (BodyTrans k a, BodySize a, HVectStackable k b c) => HVectStackable k a (b ': c) where
  type HVectSpecToCom k a (b ': c) = Stack (SpecToCom k a) (HVectSpecToCom k b c)
  hvectTrans bk sz (a :&: bc) = do c0 <- btrans bk sz a
                                   cs <- hvectTrans bk (bsize sz a) bc
                                   return (Stack c0 cs)

instance (HVectStackable k s0 ss) => BodyTrans k (HVect (s0 ': ss)) where
  type SpecToCom k (HVect (s0 ': ss)) = HVectSpecToCom k s0 ss
  btrans bk sz spec = hvectTrans bk sz spec

instance BodyTrans k (HVect '[]) where
  type SpecToCom k (HVect '[]) = TypeError (Text "HVect '[] is not a valid specification of Neural Network")
  btrans bk sz spec = undefined

class HVectSize a b where
  hvectSize  :: LayerSize -> HVect (a ': b) -> LayerSize

instance BodySize a => HVectSize a '[] where
  hvectSize  sz (a :&: HNil) = bsize sz a

instance (BodySize a, HVectSize b c) => HVectSize a (b ': c) where
  hvectSize  sz (a :&: bc) = hvectSize (bsize sz a) bc

instance (HVectSize s0 ss) => BodySize (HVect (s0 ': ss)) where
  bsize s spec = hvectSize s spec
