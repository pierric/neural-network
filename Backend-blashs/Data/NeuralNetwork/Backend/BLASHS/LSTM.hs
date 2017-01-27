------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Backend.BLASHS.LSTM
-- Description :  A backend for neuralnetwork with blas-hs.
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module supplies a LSTM component.
------------------------------------------------------------
{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts, GADTs #-}
{-# LANGUAGE DeriveDataTypeable #-}
module Data.NeuralNetwork.Backend.BLASHS.LSTM(
  LSTM(..)
) where

import Control.Monad.State.Strict
import Data.Foldable (foldrM)
import qualified Data.Vector as V
import qualified Data.Map as M
import Data.Data
import Data.Generics
import Prelude hiding (tanh)
import Data.NeuralNetwork
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD

type VecR = DenseVector Float
type MatR = DenseMatrix Float
type LSTMident = Int

data LSTM = LLSTM { parm_w_f :: MatR, parm_w_i :: MatR, parm_w_o :: MatR, parm_w_c :: MatR
                  , parm_u_f :: MatR, parm_u_i :: MatR, parm_u_o :: MatR
                  , parm_b_f :: VecR, parm_b_i :: VecR, parm_b_o :: VecR, parm_b_c :: VecR
                  , lstm_id  :: LSTMident
                  }
  deriving Typeable

instance Data LSTM where
  toConstr a = lstmConstr
  gfoldl f z a = z (\i->a{lstm_id=i}) `f` (lstm_id a)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(LSTM)"
  dataTypeOf _  = lstmDataType

lstmConstr = mkConstr lstmDataType "LSTM" ["identifier"] Prefix
lstmDataType = mkDataType "Data.NeuralNetwork.Backend.BLASHS.LSTM.LSTM" [lstmConstr]

-- state passed forward
type LSTMstreamPrev = VecR
-- state passed backward
data LSTMstreamNext = NextNothing
                    | NextJust { nx_mf, nx_mi, nx_mo, nx_f,
                                 nx_delta_c, nx_delta_f, nx_delta_i, nx_delta_o :: VecR }
-- sum-type of the forward and backward state
type LSTMstreamInfo = Either LSTMstreamPrev LSTMstreamNext

instance Component LSTM where
  -- The state is mapping from LSTM identifier to Info.
  -- So when mutiple LSTM compoents are stacked, each can
  -- access its own state.
  type Run LSTM = StateT (M.Map LSTMident LSTMstreamInfo) IO
  type Inp LSTM = VecR
  type Out LSTM = VecR
  data Trace LSTM = LTrace { tr_mf, tr_mi, tr_mo, tr_n, tr_f, tr_i, tr_o, tr_c', tr_c, tr_inp, tr_out :: VecR }
  forwardT lstm x_t = do
    Just (Left c_tm1) <- gets (M.lookup $ lstm_id lstm)
    mf_t  <- newDenseVector (size x_t)
    mf_t <<= x_t   :<# parm_w_f lstm
    mf_t <<+ c_tm1 :<# parm_u_f lstm
    mf_t <<= mf_t  :.+ parm_b_f lstm
    f_t   <- newDenseVectorCopy mf_t
    f_t  <<= Apply sigma

    mi_t <- newDenseVector (size x_t)
    mi_t <<= x_t   :<# parm_w_i lstm
    mi_t <<+ c_tm1 :<# parm_u_i lstm
    mi_t <<= mi_t  :.+ parm_b_i lstm
    i_t   <- newDenseVectorCopy mi_t
    i_t  <<= Apply sigma

    mo_t <- newDenseVector (size x_t)
    mo_t <<= x_t   :<# parm_w_o lstm
    mo_t <<+ c_tm1 :<# parm_u_o lstm
    mo_t <<= mo_t  :.+ parm_b_o lstm
    o_t   <- newDenseVectorCopy mo_t
    o_t  <<= Apply sigma

    n_t  <- newDenseVector (size x_t)
    n_t  <<= x_t :<# parm_w_c lstm
    n_t  <<= n_t :.+ parm_b_c lstm

    c_t   <- newDenseVector (size c_tm1)
    c_t  <<= c_tm1 :.* f_t

    tmp   <- newDenseVectorCopy n_t
    tmp  <<= Apply sigma
    tmp  <<= i_t :.* tmp
    c_t  <<= c_t :.+ tmp

    denseVectorCopy tmp c_t
    tmp <<= Apply sigma
    tmp <<= tmp :.* o_t

    modify $ M.insert (lstm_id lstm) (Left c_t)

    let trace = LTrace { tr_mf = mf_t, tr_mi = mi_t, tr_mo = mo_t
                       , tr_n = n_t
                       , tr_f = f_t, tr_i = i_t, tr_o = o_t
                       , tr_c' = c_tm1, tr_c = c_t
                       , tr_inp = x_t, tr_out = tmp }
    return $ trace

  output = tr_out

  backward lstm trace !delta_out rate = do
    Just (Right upward) <- gets (M.lookup $ lstm_id lstm)
    delta_ct <- case upward of
                  NextNothing -> do
                    tmp <- newDenseVectorCopy (tr_c trace)
                    tmp <<= Apply sigma'
                    tmp <<= tmp :.* tr_o trace
                    tmp <<= tmp :.* delta_out
                    return tmp
                  nx -> do
                    tmp <- newDenseVectorCopy (tr_c trace)
                    tmp <<= nx_f nx :.* nx_delta_c nx

                    -- NOTE: c_t, mf_(t+1), mi_(t+1), mo_(t+1) shall not used any more
                    c_t <- newDenseVectorCopy (tr_c trace)
                    c_t <<= Apply sigma'
                    c_t <<= c_t :.* tr_o trace
                    c_t <<= c_t :.* delta_out
                    tmp <<= tmp :.+ c_t

                    mf_tp1 <- newDenseVectorCopy (nx_mf nx)
                    mf_tp1 <<= Apply sigma'
                    mf_tp1 <<= parm_u_f lstm :#> mf_tp1
                    mf_tp1 <<= mf_tp1 :.* nx_delta_f nx
                    tmp <<= tmp :.+ mf_tp1

                    mi_tp1 <- newDenseVectorCopy (nx_mi nx)
                    mi_tp1 <<= Apply sigma'
                    mi_tp1 <<= parm_u_i lstm :#> mi_tp1
                    mi_tp1 <<= mi_tp1 :.* nx_delta_i nx
                    tmp <<= tmp :.+ mi_tp1

                    mo_tp1 <- newDenseVectorCopy (nx_mo nx)
                    mo_tp1 <<= Apply sigma'
                    mo_tp1 <<= parm_u_o lstm :#> mo_tp1
                    mo_tp1 <<= mo_tp1 :.* nx_delta_o nx
                    tmp <<= tmp :.+ mo_tp1

                    return tmp

    delta_ft  <- newDenseVector (size delta_ct)
    delta_ft <<= tr_c' trace :.* delta_ct

    delta_it  <- newDenseVectorCopy (tr_n trace)
    delta_it <<= Apply sigma
    delta_it <<= delta_it :.* delta_ct

    delta_ot  <- newDenseVectorCopy (tr_c trace)
    delta_ot <<= Apply sigma
    delta_ot <<= delta_ot :.* delta_out

    delta_bc  <- newDenseVectorCopy (tr_n trace)
    delta_bc <<= Apply sigma'
    delta_bc <<= delta_bc :.* tr_inp trace
    delta_bc <<= delta_bc :.* delta_ct

    delta_bf  <- newDenseVectorCopy (tr_mf trace)
    delta_bf <<= Apply sigma'
    delta_bf <<= delta_bf :.* delta_ft

    delta_bi  <- newDenseVectorCopy (tr_mi trace)
    delta_bi <<= Apply sigma'
    delta_bi <<= delta_bi :.* delta_it

    delta_bo  <- newDenseVectorCopy (tr_mo trace)
    delta_bo <<= Apply sigma'
    delta_bo <<= delta_bo :.* delta_ot

    delta_wc <- uncurry newDenseMatrix (size (parm_w_c lstm))
    tmp <- newDenseVectorCopy (tr_n trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* tr_i trace
    tmp <<= tmp :.* delta_ct
    delta_wc <<= tmp :## tr_inp trace

    delta_wf <- uncurry newDenseMatrix (size (parm_w_f lstm))
    denseVectorCopy tmp (tr_mf trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ft
    delta_wf <<= tmp :## tr_inp trace

    delta_wi <- uncurry newDenseMatrix (size (parm_w_i lstm))
    denseVectorCopy tmp (tr_mi trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_it
    delta_wi <<= tmp :## tr_inp trace

    delta_wo <- uncurry newDenseMatrix (size (parm_w_o lstm))
    denseVectorCopy tmp (tr_mo trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ot
    delta_wo <<= tmp :## tr_inp trace

    delta_uf <- uncurry newDenseMatrix (size (parm_u_f lstm))
    denseVectorCopy tmp (tr_mf trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ft
    delta_uf <<= tmp :## tr_c' trace

    delta_ui <- uncurry newDenseMatrix (size (parm_u_i lstm))
    denseVectorCopy tmp (tr_mi trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_it
    delta_ui <<= tmp :## tr_c' trace

    delta_uo <- uncurry newDenseMatrix (size (parm_u_o lstm))
    denseVectorCopy tmp (tr_mo trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ot
    delta_uo <<= tmp :## tr_c' trace

    delta_inp <- newDenseVector (size delta_out)
    tmp <- newDenseVectorCopy (tr_mf trace)
    tmp <<= Apply sigma'
    tmp <<= parm_w_f lstm :#> tmp
    delta_inp <<= tmp :.* delta_ft

    denseVectorCopy tmp (tr_mi trace)
    tmp <<= Apply sigma'
    tmp <<= parm_w_i lstm :#> tmp
    delta_inp <<+ tmp :.* delta_it

    denseVectorCopy tmp (tr_mo trace)
    tmp <<= Apply sigma'
    tmp <<= parm_w_o lstm :#> tmp
    delta_inp <<+ tmp :.* delta_ot

    denseVectorCopy tmp (tr_n trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* tr_i trace
    tmp <<= parm_w_c lstm :#> tmp
    delta_inp <<+ tmp :.* delta_ct

    delta_bc <<= Scale rate
    parm_b_c lstm <<= parm_b_c lstm :.+ delta_bc
    delta_bf <<= Scale rate
    parm_b_f lstm <<= parm_b_f lstm :.+ delta_bf
    delta_bi <<= Scale rate
    parm_b_i lstm <<= parm_b_i lstm :.+ delta_bi
    delta_bo <<= Scale rate
    parm_b_o lstm <<= parm_b_o lstm :.+ delta_bo
    delta_wc <<= Scale rate
    parm_w_c lstm <<= parm_w_c lstm :.+ delta_wc
    delta_wf <<= Scale rate
    parm_w_f lstm <<= parm_w_f lstm :.+ delta_wf
    delta_wi <<= Scale rate
    parm_w_i lstm <<= parm_w_i lstm :.+ delta_wi
    delta_wo <<= Scale rate
    parm_w_o lstm <<= parm_w_o lstm :.+ delta_wo
    delta_uf <<= Scale rate
    parm_u_f  lstm<<= parm_u_f lstm :.+ delta_uf
    delta_ui <<= Scale rate
    parm_u_i lstm <<= parm_u_i lstm :.+ delta_ui
    delta_uo <<= Scale rate
    parm_u_o lstm <<= parm_u_o lstm :.+ delta_uo

    modify $ M.insert (lstm_id lstm)
              (Right $ NextJust {
                nx_mf = tr_mf trace,
                nx_mi = tr_mi trace,
                nx_mo = tr_mo trace,
                nx_f  = tr_f  trace,
                nx_delta_c = delta_ct,
                nx_delta_f = delta_ft,
                nx_delta_i = delta_it,
                nx_delta_o = delta_ot
              })
    return (lstm, delta_inp)

newtype Stream a = Stream a
  deriving (Typeable, Data)

instance (Data a, Component a, Inp a ~ VecR, Run a ~ Run LSTM) => Component (Stream a) where
  type Run (Stream a) = IO
  type Inp (Stream a) = [Inp a]
  type Out (Stream a) = [Out a]
  newtype Trace (Stream a) = StreamTrace [Trace a]
  forwardT (Stream c) xs = do
    -- set initial state for all LSTMs
    st <- forM (collectLSTMs c) (\lstm -> do
            vec <- newDenseVector (size $ head xs)
            return (lstm_id lstm, Left vec))
    -- forward each input one by one, where the state is implicitly propagated.
    trs <- flip evalStateT (M.fromList st) (mapM (forwardT c) xs)
    return $ StreamTrace trs
  output (StreamTrace trace) = map output trace
  backward (Stream c) (StreamTrace trace) delta_out rate = do
    -- set initial state for all LSTMs
    st <- forM (collectLSTMs c) (\lstm ->
            return (lstm_id lstm, Right NextNothing))
    -- backward for each input one by one, and accumulate all updates
    (c, delta_inp) <- flip evalStateT (M.fromList st) $ foldrM step (c, []) (zip trace delta_out)
    return (Stream c, delta_inp)
    where
      step (tr,dout) (c,ds) = do
        (c', di) <- backward c tr dout rate
        return (c', di:ds)

collectLSTMs :: Data a => a -> [LSTM]
collectLSTMs = everything (++) ([] `mkQ` isLSTM)
  where isLSTM a@(LLSTM{}) = [a]

sigma  = tanh
sigma' = tanh'

tanh  x = let x2 = times x x
              x3 = times x x2
          in minus x (divide x3 (konst 3))
tanh' x = let a = tanh x
              b = times a a
          in minus (konst 1) b
