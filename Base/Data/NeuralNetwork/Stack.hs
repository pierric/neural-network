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
{-# LANGUAGE ConstraintKinds #-}
module Data.NeuralNetwork.Stack (
  Stack(..), Lift(..)
) where

import Data.Data
import Data.HVect
import GHC.TypeLits
import Data.Constraint (Dict(..), withDict, Constraint)
import Control.Monad.Trans
import Control.Monad.Except (MonadError)
import Data.NeuralNetwork

data Stack (a :: * -> *) (b :: * -> *) o = Stack (a o) (b o) (Dict (Data (a o)), Dict (Data (b o)))
  deriving Typeable

instance (Typeable a, Typeable b, Typeable o, Data (a o), Data (b o)) => Data (Stack a b o) where
  toConstr a = stackConstr
  gfoldl f z (Stack a b e) = z Stack `f` a `f` b `f` e
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(Stack)"
  dataTypeOf _  = stackDataType

stackConstr = mkConstr stackDataType "Stack" ["a", "b"] Prefix
stackDataType = mkDataType "Data.NeuralNetwork.Stack.Stack" [stackConstr]

instance (Component a, Component b,
          Monad (Run a), Monad (Run b),
          Run a ~ Run b,
          Out a ~ Inp b
         ) => Component (Stack a b) where
  type Dty (Stack a b) = Dty a
  type Run (Stack a b) = Run a
  type Inp (Stack a b) = Inp a
  type Out (Stack a b) = Out b
  newtype Trace (Stack a b) = S0Trace (Trace b, Trace a)
  forwardT (Stack a b e) !i =
    case e of
      (Dict, Dict) -> do
        !tra <- forwardT a i
        !trb <- forwardT b (output tra)
        return $ S0Trace (trb, tra)
  output (S0Trace !a) = output (fst a)
  backward (Stack a b e) (S0Trace (!trb,!tra)) !odeltb =
    case e of
      (Dict, Dict) -> do
        (b', !odelta) <- backward b trb odeltb
        (a', !idelta) <- backward a tra odelta
        return (Stack a' b' e, idelta)

data Lift (t :: (* -> *) -> * -> *) (c :: * -> *) o = Lift (c o) (Dict (Data (c o)))
  deriving (Typeable, Data)
instance (Component c, Typeable t, MonadTrans t, Monad (t (Run c))) => Component (Lift t c) where
  type Dty (Lift t c) = Dty c
  type Run (Lift t c) = t (Run c)
  type Inp (Lift t c) = Inp c
  type Out (Lift t c) = Out c
  newtype Trace (Lift t c) = LtTrace (Trace c)
  forwardT (Lift c e) !i = withDict e $ do
    o <- lift $ forwardT c i
    return $ LtTrace o
  output (LtTrace !a) = output a
  backward (Lift c e) (LtTrace t) !odeltb = withDict e $ do
    (c', idelta) <- lift $ backward c t odeltb
    return (Lift c' e, idelta)

-- internal type class to do induction on a non-empty hvect
class MonadError ErrCode (Env k) => HVectStackable k a b where
  type HVectSpecToCom k a b :: * -> *
  hvectWitness :: Optimizer o => k -> o -> HVect (a ': b) -> Dict (Component (HVectSpecToCom k a b), Data (HVectSpecToCom k a b o))
  hvectTrans   :: (Optimizer o, Optimizable k o) => k -> o -> LayerSize -> HVect (a ': b) -> Env k (HVectSpecToCom k a b o)

instance BodyTrans k a => HVectStackable k a '[] where
  type HVectSpecToCom k a '[] = SpecToCom k a
  hvectTrans bk o sz (a :&: HNil) = btrans bk o sz a
  hvectWitness bk o  (a :&: HNil) = bwitness bk o a

instance (BodyTrans k a, BodySize a, HVectStackable k b c,
          Out (SpecToCom k a) ~ Inp (HVectSpecToCom k b c),
          Run (SpecToCom k a) ~ Run (HVectSpecToCom k b c)
          ) => HVectStackable k a (b ': c) where
  type HVectSpecToCom k a (b ': c) = Stack (SpecToCom k a) (HVectSpecToCom k b c)
  hvectTrans bk o sz (a :&: bc) = do c0 <- btrans bk o sz a
                                     cs <- hvectTrans bk o (bsize sz a) bc
                                     case (bwitness bk o a, hvectWitness bk o bc) of
                                       (Dict, Dict) -> return (Stack c0 cs (Dict, Dict))
  hvectWitness bk o (a :&: bc)  = case (bwitness bk o a, hvectWitness bk o bc) of
                                    (Dict, Dict) -> Dict

instance HVectStackable k s0 ss => BodyTrans k (HVect (s0 ': ss)) where
  type SpecToCom k (HVect (s0 ': ss)) = HVectSpecToCom k s0 ss
  btrans bk o sz spec = hvectTrans bk o sz spec
  bwitness = hvectWitness

instance MonadError ErrCode (Env k) => BodyTrans k (HVect '[]) where
  type SpecToCom k (HVect '[]) = TypeError (Text "HVect '[] is not a valid specification of Neural Network")
  bwitness = undefined
  btrans   = undefined

class HVectSize a b where
  hvectSize  :: LayerSize -> HVect (a ': b) -> LayerSize

instance BodySize a => HVectSize a '[] where
  hvectSize  sz (a :&: HNil) = bsize sz a

instance (BodySize a, HVectSize b c) => HVectSize a (b ': c) where
  hvectSize  sz (a :&: bc) = hvectSize (bsize sz a) bc

instance (HVectSize s0 ss) => BodySize (HVect (s0 ': ss)) where
  bsize s spec = hvectSize s spec
