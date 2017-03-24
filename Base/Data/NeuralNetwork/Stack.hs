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
  Stack(..),
  CE, CL, CR, LiftRun,
) where

import Data.Data
import Data.HVect
import GHC.TypeLits
import Data.Constraint (Dict(..))
import Control.Monad.Trans
import Control.Monad.Except (MonadError)
import Data.NeuralNetwork

data Stack (a :: * -> *) (b :: * -> *) c o = Stack (a o) (b o) (Dict (Data (a o)), Dict (Data (b o)))
  deriving Typeable

instance (Typeable a, Typeable b, Typeable c, Typeable o, Data (a o), Data (b o)) => Data (Stack a b c o) where
  toConstr a = stackConstr
  gfoldl f z (Stack a b e) = z Stack `f` a `f` b `f` e
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(Stack)"
  dataTypeOf _  = stackDataType

stackConstr = mkConstr stackDataType "Stack" ["a", "b"] Prefix
stackDataType = mkDataType "Data.NeuralNetwork.Stack.Stack" [stackConstr]

data CE
data CL (t :: (* -> *) -> * -> *)
data CR (t :: (* -> *) -> * -> *)
type family LiftRun (u :: * -> *) (v :: * -> *) where
  LiftRun (t m) m = CL t
  LiftRun m (t m) = CR t
  LiftRun m m     = CE

instance (Component a, Component b,
          Monad (Run a), Monad (Run b),
          Run a ~ Run b,
          Out a ~ Inp b
         ) => Component (Stack a b CE) where
  type Dty (Stack a b CE) = Dty a
  type Run (Stack a b CE) = Run a
  type Inp (Stack a b CE) = Inp a
  type Out (Stack a b CE) = Out b
  newtype Trace (Stack a b CE) = S0Trace (Trace b, Trace a)
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

instance (Component a, Component b, Typeable t,
          Monad (Run a), Monad (Run b),
          MonadTrans t, Run a ~ t (Run b),
          Out a ~ Inp b
         ) => Component (Stack a b (CL t)) where
  type Dty (Stack a b (CL t)) = Dty a
  type Run (Stack a b (CL t)) = Run a
  type Inp (Stack a b (CL t)) = Inp a
  type Out (Stack a b (CL t)) = Out b
  newtype Trace (Stack a b (CL t)) = S1Trace (Trace b, Trace a)
  forwardT (Stack a b e) !i = do
    case e of
      (Dict, Dict) -> do
        !tra <- forwardT a i
        !trb <- lift $ forwardT b (output tra)
        return $ S1Trace (trb, tra)
  output (S1Trace !a) = output (fst a)
  backward (Stack a b e) (S1Trace (!trb,!tra)) !odeltb = do
    case e of
      (Dict, Dict) -> do
        (b', !odelta) <- lift $ backward b trb odeltb
        (a', !idelta) <- backward a tra odelta
        return (Stack a' b' e, idelta)

instance (Component a, Component b, Typeable t,
          Monad (Run a), Monad (Run b),
          MonadTrans t, Run b ~ t (Run a),
          Out a ~ Inp b
         ) => Component (Stack a b (CR t)) where
  type Dty (Stack a b (CR t)) = Dty a
  type Run (Stack a b (CR t)) = Run b
  type Inp (Stack a b (CR t)) = Inp a
  type Out (Stack a b (CR t)) = Out b
  newtype Trace (Stack a b (CR t)) = S2Trace (Trace b, Trace a)
  forwardT (Stack a b e) !i = do
    case e of
      (Dict, Dict) -> do
        !tra <- lift $ forwardT a i
        !trb <- forwardT b (output tra)
        return $ S2Trace (trb, tra)
  output (S2Trace !a) = output (fst a)
  backward (Stack a b e) (S2Trace (!trb,!tra)) !odeltb = do
    case e of
      (Dict, Dict) -> do
        (b', !odelta) <- backward b trb odeltb
        (a', !idelta) <- lift $ backward a tra odelta
        return (Stack a' b' e, idelta)

-- internal type class to do induction on a non-empty hvect
class MonadError ErrCode (Env k) => HVectStackable k a b where
  type HVectSpecToCom k a b :: * -> *
  hvectWitness :: Optimizer o => k -> HVect (a ': b) -> o -> Dict (Component (HVectSpecToCom k a b), Data (HVectSpecToCom k a b o), RunInEnv (Run (HVectSpecToCom k a b)) (Env k))
  hvectTrans   :: Optimizer o => k -> LayerSize -> HVect (a ': b) -> o -> Env k (HVectSpecToCom k a b o)

instance BodyTrans k a => HVectStackable k a '[] where
  type HVectSpecToCom k a '[] = SpecToCom k a
  hvectTrans bk sz (a :&: HNil) o = btrans bk sz a o

instance (BodyTrans k a, BodySize a, HVectStackable k b c) => HVectStackable k a (b ': c) where
  type HVectSpecToCom k a (b ': c) = Stack (SpecToCom k a) (HVectSpecToCom k b c) (LiftRun (Run (SpecToCom k a)) (Run (HVectSpecToCom k b c)))
  hvectTrans bk sz (a :&: bc) o = do c0 <- btrans bk sz a o
                                     cs <- hvectTrans bk (bsize sz a) bc o
                                     case (bwitness bk a o, hvectWitness bk bc o) of
                                       (Dict, Dict) -> return (Stack c0 cs (Dict, Dict))

instance HVectStackable k s0 ss => BodyTrans k (HVect (s0 ': ss)) where
  type SpecToCom k (HVect (s0 ': ss)) = HVectSpecToCom k s0 ss
  btrans bk sz spec o = hvectTrans bk sz spec o

instance MonadError ErrCode (Env k) => BodyTrans k (HVect '[]) where
  type SpecToCom k (HVect '[]) = TypeError (Text "HVect '[] is not a valid specification of Neural Network")

class HVectSize a b where
  hvectSize  :: LayerSize -> HVect (a ': b) -> LayerSize

instance BodySize a => HVectSize a '[] where
  hvectSize  sz (a :&: HNil) = bsize sz a

instance (BodySize a, HVectSize b c) => HVectSize a (b ': c) where
  hvectSize  sz (a :&: bc) = hvectSize (bsize sz a) bc

instance (HVectSize s0 ss) => BodySize (HVect (s0 ': ss)) where
  bsize s spec = hvectSize s spec
