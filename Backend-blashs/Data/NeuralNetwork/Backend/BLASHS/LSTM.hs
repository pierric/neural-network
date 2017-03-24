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
{-# LANGUAGE UndecidableInstances #-}
module Data.NeuralNetwork.Backend.BLASHS.LSTM(
  LSTM(..), LSTM_Env_Transformer, newLSTM, Stream(..),
) where

import Blas.Generic.Unsafe (Numeric)
import Control.Monad.State.Strict
import Data.Foldable (foldrM)
import qualified Data.Vector as V
import qualified Data.Map as M
import Data.Data
import Data.Generics
import Data.Constraint (Dict(..), withDict)
import Data.IORef
import System.IO.Unsafe (unsafePerformIO)
import Prelude hiding (tanh)
import Data.NeuralNetwork
import Data.NeuralNetwork.Adapter
import Data.NeuralNetwork.Backend.BLASHS.Utils
import Data.NeuralNetwork.Backend.BLASHS.SIMD
import System.Random.MWC
import System.Random.MWC.Distributions
-- import GHC.Float (double2Float)
import Data.Vector.Storable (Storable)
import qualified Text.PrettyPrint.Free as P

type VecR = DenseVector
type MatR = DenseMatrix

type LSTMident = Int

data LSTM p o = LLSTM { parm_w_f, parm_w_i, parm_w_o, parm_w_c, parm_u_f, parm_u_i, parm_u_o :: WithVar MatR o p
                      , parm_b_f, parm_b_i, parm_b_o, parm_b_c :: WithVar VecR o p
                      , lstm_id  :: LSTMident, lstm_isize, lstm_osize :: Int
                      , lstm_opt_ev :: (Dict (Optimizable o (VecR p)), Dict (Optimizable o (MatR p)))
                      }
  deriving Typeable


instance (Data o, Data p) => Data (LSTM p o) where
  toConstr a = lstmConstr
  gfoldl f z a = z (\i->a{lstm_id=i}) `f` (lstm_id a)
  gunfold k z c = errorWithoutStackTrace "Data.Data.gunfold(LSTM)"
  dataTypeOf _  = lstmDataType

lstmConstr = mkConstr lstmDataType "LSTM" ["identifier"] Prefix
lstmDataType = mkDataType "Data.NeuralNetwork.Backend.BLASHS.LSTM.LSTM" [lstmConstr]

global_LSTM_id_counter :: IORef Int
global_LSTM_id_counter = unsafePerformIO (newIORef 0)
-- | create a new LSTM component
newLSTM :: (Precision p, Optimizer o, Optimizable o (VecR p), Optimizable o (MatR p))
        => Int        -- ^ input size
        -> Int        -- ^ output size
        -> o
        -> IO (LSTM p o) -- ^ the new layer
newLSTM m n opt =
  withSystemRandom . asGenIO $ \gen -> do
    let newW v = do raw <- newDenseVectorByGen (fromDouble <$> normal 0 v gen) (m*n)
                    let mw = v2m m n raw
                    ov  <- newOptVar opt mw
                    return $ WithVar mw ov
        newU v = do raw <- newDenseVectorByGen (fromDouble <$> normal 0 v gen) (n*n)
                    let mu = v2m n n raw
                    ov  <- newOptVar opt mu
                    return $ WithVar mu ov
        newB v = do vb  <- newDenseVectorConst n v
                    ov  <- newOptVar opt vb
                    return $ WithVar vb ov
    parm_w_f <- newW 0.01
    parm_w_i <- newW 0.01
    parm_w_o <- newW 0.01
    parm_w_c <- newW 0.01
    parm_u_f <- newU 0.01
    parm_u_i <- newU 0.01
    parm_u_o <- newU 0.01
    parm_b_f <- newB 1.0
    parm_b_i <- newB 0
    parm_b_o <- newB 0
    parm_b_c <- newB 0
    lstm_id  <- readIORef global_LSTM_id_counter
    modifyIORef' global_LSTM_id_counter (+1)
    return $ LLSTM {
      parm_w_f = parm_w_f,
      parm_w_i = parm_w_i,
      parm_w_o = parm_w_o,
      parm_w_c = parm_w_c,
      parm_u_f = parm_u_f,
      parm_u_i = parm_u_i,
      parm_u_o = parm_u_o,
      parm_b_f = parm_b_f,
      parm_b_i = parm_b_i,
      parm_b_o = parm_b_o,
      parm_b_c = parm_b_c,
      lstm_id  = lstm_id,
      lstm_isize = m,
      lstm_osize = n,
      lstm_opt_ev = (Dict, Dict)
    }

-- state passed forward
type LSTMstreamPrev p = VecR p
-- state passed backward
data LSTMstreamNext p = NextNothing
                      | NextJust { nx_mf, nx_mi, nx_mo, nx_f,
                                   nx_delta_c, nx_delta_f, nx_delta_i, nx_delta_o :: VecR p,
                                   nx_ori_uf, nx_ori_ui, nx_ori_uo :: MatR p }
-- sum-type of the forward and backward state
type LSTMstreamInfo p = Either (LSTMstreamPrev p) (LSTMstreamNext p)

type LSTM_Env_Transformer p = StateT (M.Map LSTMident (LSTMstreamInfo p))

instance (Typeable p, Numeric p, RealType p, SIMDable p) => Component (LSTM p) where
  -- The state is mapping from LSTM identifier to Info.
  -- So when mutiple LSTM compoents are stacked, each can
  -- access its own state.
  type Dty (LSTM p) = p
  type Run (LSTM p) = LSTM_Env_Transformer p IO
  type Inp (LSTM p) = VecR p
  type Out (LSTM p) = VecR p
  data Trace (LSTM p) = LTrace { tr_mf, tr_mi, tr_mo, tr_n, tr_f, tr_i, tr_o, tr_c', tr_c, tr_inp, tr_out :: VecR p }
  forwardT lstm x_t = do
    Just (Left c_tm1) <- gets (M.lookup $ lstm_id lstm)

    let osize = lstm_osize lstm
    mf_t  <- newDenseVector osize
    mf_t <<= x_t   :<# _parm (parm_w_f lstm)
    mf_t <<+ c_tm1 :<# _parm (parm_u_f lstm)
    mf_t <<= mf_t  :.+ _parm (parm_b_f lstm)
    f_t   <- newDenseVectorCopy mf_t
    f_t  <<= Apply sigma

    mi_t <- newDenseVector osize
    mi_t <<= x_t   :<# _parm (parm_w_i lstm)
    mi_t <<+ c_tm1 :<# _parm (parm_u_i lstm)
    mi_t <<= mi_t  :.+ _parm (parm_b_i lstm)
    i_t   <- newDenseVectorCopy mi_t
    i_t  <<= Apply sigma

    mo_t <- newDenseVector osize
    mo_t <<= x_t   :<# _parm (parm_w_o lstm)
    mo_t <<+ c_tm1 :<# _parm (parm_u_o lstm)
    mo_t <<= mo_t  :.+ _parm (parm_b_o lstm)
    o_t   <- newDenseVectorCopy mo_t
    o_t  <<= Apply sigma

    n_t  <- newDenseVector osize
    n_t  <<= x_t :<# _parm (parm_w_c lstm)
    n_t  <<= n_t :.+ _parm (parm_b_c lstm)

    c_t   <- newDenseVector osize
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

  backward lstm trace !delta_out = withDict (fst $ lstm_opt_ev lstm) $ withDict (snd $ lstm_opt_ev lstm) $ do
    Just (Right upward) <- gets (M.lookup $ lstm_id lstm)

    (delta_ct, ori_uf, ori_ui, ori_uo) <- case upward of
                  NextNothing -> do
                    tmp <- newDenseVectorCopy (tr_c trace)
                    tmp <<= Apply sigma'
                    tmp <<= tmp :.* tr_o trace
                    tmp <<= tmp :.* delta_out

                    -- the original Ui, Uf, Uo are used in calc delta_ct,
                    -- so save a copy in state.
                    ori_uf <- newDenseMatrixCopy (_parm $ parm_u_f lstm)
                    ori_ui <- newDenseMatrixCopy (_parm $ parm_u_i lstm)
                    ori_uo <- newDenseMatrixCopy (_parm $ parm_u_o lstm)
                    return (tmp, ori_uf, ori_ui, ori_uo)
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
                    mf_tp1 <<= nx_ori_uf nx :#> mf_tp1
                    mf_tp1 <<= mf_tp1 :.* nx_delta_f nx
                    tmp <<= tmp :.+ mf_tp1

                    mi_tp1 <- newDenseVectorCopy (nx_mi nx)
                    mi_tp1 <<= Apply sigma'
                    mi_tp1 <<= nx_ori_ui nx :#> mi_tp1
                    mi_tp1 <<= mi_tp1 :.* nx_delta_i nx
                    tmp <<= tmp :.+ mi_tp1

                    mo_tp1 <- newDenseVectorCopy (nx_mo nx)
                    mo_tp1 <<= Apply sigma'
                    mo_tp1 <<= nx_ori_uo nx :#> mo_tp1
                    mo_tp1 <<= mo_tp1 :.* nx_delta_o nx
                    tmp <<= tmp :.+ mo_tp1

                    return (tmp, nx_ori_uf nx, nx_ori_ui nx, nx_ori_uo nx)

    delta_ft  <- newDenseVector (lstm_osize lstm)
    delta_ft <<= tr_c' trace :.* delta_ct

    delta_it  <- newDenseVectorCopy (tr_n trace)
    delta_it <<= Apply sigma
    delta_it <<= delta_it :.* delta_ct

    delta_ot  <- newDenseVectorCopy (tr_c trace)
    delta_ot <<= Apply sigma
    delta_ot <<= delta_ot :.* delta_out

    delta_bc  <- newDenseVectorCopy (tr_n trace)
    delta_bc <<= Apply sigma'
    delta_bc <<= delta_bc :.* tr_i trace
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

    let mat_size = (lstm_isize lstm, lstm_osize lstm)
    delta_wc <- uncurry newDenseMatrix mat_size
    tmp <- newDenseVectorCopy (tr_n trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* tr_i trace
    tmp <<= tmp :.* delta_ct
    delta_wc <<+ tr_inp trace :## tmp

    delta_wf <- uncurry newDenseMatrix mat_size
    denseVectorCopy tmp (tr_mf trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ft
    delta_wf <<+ tr_inp trace :## tmp

    delta_wi <- uncurry newDenseMatrix mat_size
    denseVectorCopy tmp (tr_mi trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_it
    delta_wi <<+ tr_inp trace :## tmp

    delta_wo <- uncurry newDenseMatrix mat_size
    denseVectorCopy tmp (tr_mo trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ot
    delta_wo <<+ tr_inp trace :## tmp

    delta_uf <- uncurry newDenseMatrix mat_size
    denseVectorCopy tmp (tr_mf trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ft
    delta_uf <<+ tr_c' trace :## tmp

    delta_ui <- uncurry newDenseMatrix mat_size
    denseVectorCopy tmp (tr_mi trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_it
    delta_ui <<+ tr_c' trace :## tmp

    delta_uo <- uncurry newDenseMatrix mat_size
    denseVectorCopy tmp (tr_mo trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ot
    delta_uo <<+ tr_c' trace :## tmp

    delta_inp <- newDenseVector (lstm_isize lstm)
    tmp <- newDenseVectorCopy (tr_mf trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ft
    delta_inp <<= _parm (parm_w_f lstm) :#> tmp

    denseVectorCopy tmp (tr_mi trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_it
    delta_inp <<+ _parm (parm_w_i lstm) :#> tmp

    denseVectorCopy tmp (tr_mo trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* delta_ot
    delta_inp <<+ _parm (parm_w_o lstm) :#> tmp

    denseVectorCopy tmp (tr_n trace)
    tmp <<= Apply sigma'
    tmp <<= tmp :.* tr_i trace
    tmp <<= tmp :.* delta_ct
    delta_inp <<+ _parm (parm_w_c lstm) :#> tmp

    let parms = [parm_b_c,parm_b_f,parm_b_i,parm_b_o]
        upds  = [delta_bc,delta_bf,delta_bi,delta_bo]
    forM_ (zip parms upds) $ \(p, u) -> do
      u <- liftIO $ optimize (_ovar $ p lstm) u
      _parm (p lstm) <<= _parm (p lstm) :.+ u

    let parms = [parm_w_c,parm_w_f,parm_w_i,parm_w_o,parm_u_f,parm_u_i,parm_u_o]
        upds  = [delta_wc,delta_wf,delta_wi,delta_wo,delta_uf,delta_ui,delta_uo]
    forM_ (zip parms upds) $ \(p, u) -> do
      u <- liftIO $ optimize (_ovar $ p lstm) u
      _parm (p lstm) <<= _parm (p lstm) :.+ u

    modify $ M.insert (lstm_id lstm)
              (Right $ NextJust {
                nx_mf = tr_mf trace,
                nx_mi = tr_mi trace,
                nx_mo = tr_mo trace,
                nx_f  = tr_f  trace,
                nx_delta_c = delta_ct,
                nx_delta_f = delta_ft,
                nx_delta_i = delta_it,
                nx_delta_o = delta_ot,
                nx_ori_uf  = ori_uf,
                nx_ori_ui  = ori_ui,
                nx_ori_uo  = ori_uo
              })
    return (lstm, delta_inp)

data Stream (a :: * -> *) o = Stream (a o) (Dict (Data (a o)))
  deriving (Typeable, Data)

instance (Component a, Typeable a,
          Inp a ~ VecR (Dty a),
          Typeable (Dty a), Numeric (Dty a), RealType (Dty a), SIMDable (Dty a),
          Run a ~ LSTM_Env_Transformer (Dty a) IO) => Component (Stream a) where
  type Dty (Stream a) = Dty a
  type Run (Stream a) = IO
  type Inp (Stream a) = [Inp a]
  type Out (Stream a) = [Out a]
  newtype Trace (Stream a) = StreamTrace [Trace a]
  forwardT s@(Stream c e) xs =
    withDict e $ do
        -- set initial state for all LSTMs
        st <- forM (collectLSTMs s) (\lstm -> do
                vec <- newDenseVector (lstm_osize lstm)
                return (lstm_id lstm, Left vec))
        -- forward each input one by one, where the state is implicitly propagated.
        trs <- flip evalStateT (M.fromList st) (mapM (forwardT c) xs)
        return $ StreamTrace trs
  output (StreamTrace trace) = map output trace
  backward s@(Stream c e) (StreamTrace trace) delta_out =
    withDict e $ do
        -- set initial state for all LSTMs
        st <- forM (collectLSTMs s) (\lstm ->
                return (lstm_id lstm, Right NextNothing))
        -- backward for each input one by one, and accumulate all updates
        (c, delta_inp) <- flip evalStateT (M.fromList st) $ foldrM step (c, []) (zip trace delta_out)
        return (Stream c e, delta_inp)
    where
      step (tr,dout) (c,ds) = do
        (c', di) <- backward c tr dout
        return (c', di:ds)

collectLSTMs :: (Component a, Typeable (Dty a), Typeable a, Data o, Data (a o)) => Stream a o -> [LSTM (Dty a) o]
collectLSTMs = everything (++) ([] `mkQ` isLSTM)
  where
    isLSTM a@(LLSTM{}) = [a]

sigma, sigma' :: SIMDable a => SIMDPACK a -> SIMDPACK a
sigma  = tanh
sigma' = tanh'
