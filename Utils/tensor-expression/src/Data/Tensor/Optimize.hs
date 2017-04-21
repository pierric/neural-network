{-# LANGUAGE BangPatterns, FlexibleInstances #-}
module Data.Tensor.Optimize (
  optimize
) where

import Data.Tensor.Class
import Data.Tensor.Prop
import Data.Tensor.Compile (TensorWrap(..))
import Data.Tensor.Execute
import Data.Maybe (fromJust)
import Control.Monad
import Control.Monad.Except
import Control.Monad.State.Strict
import Control.Monad.Trans.Maybe
import Data.Typeable (cast)

-- import Debug.Trace
-- import Text.PrettyPrint.Free (pretty)

type OptM = StateT CGState IO
newtype Pipeline a = Pipeline { runPipeline :: [Statement a] -> OptM [Statement a] }
-- the PipelineStep is a single step in the pipeline that
-- throws a OptError for further indication of action
data OptError a = NotEligible | Continue ![Statement a] ![Statement a]
type PipelineStep a = [Statement a] -> ExceptT (OptError a) OptM [Statement a]

instance Monoid (Pipeline a) where
  mempty = Pipeline return
  mappend (Pipeline a) (Pipeline b) = Pipeline (a >=> b)

optimize :: Element a => [Statement a] -> CG [Statement a]
optimize st = mapStateT lift $ flip runPipeline st $
  mconcat [ pipeline opt_rewrite_alloc_store_as_bind
          , pipeline opt_remove_synonym
          , pipeline opt_absorb_gemv_dotadd
          , pipeline opt_absorb_gemv_dotsca
          , pipeline opt_absorb_gemm_dotadd
          , pipeline opt_absorb_gemm_dotsca ]

pipeline :: Element a => PipelineStep a -> Pipeline a
pipeline act = Pipeline (liftM fromJust . runMaybeT . multi)
  where
    -- repeat untils the opt does not applies,
    -- return the last optimization result.
    multi st = (complete st >>= multi) `mplus` (return st)
    -- taking a sequence of statements, and complete a full
    -- cycle of the given step.
    -- produce a sequence if it applies once or else Nothing.
    complete st = do
      r <- lift $ runExceptT (act st)
      case r of
        Left NotEligible        -> mzero
        Left (Continue st1 st2) -> (st1 ++) <$> complete st2
        Right st1               -> return st1

replace (d, v1) v2 = (d, v1{_vid = _vid v2})

opt_rewrite_alloc_store_as_bind :: Element a => PipelineStep a
opt_rewrite_alloc_store_as_bind st = do
  -- liftIO $ putStrLn "opt_rewrite_alloc_store_as_bind"
  let (st1, st2) = break isAlloc st
  when (null st2) $ throwError NotEligible
  case st2 of
    Alloc d v : st3 -> do
      let (st4, st5) = break (isStoreTo $ _vid v) st3
      when (null st5) $ throwError $ Continue (st1 ++ [head st2]) (tail st2)
      case st5 of
        Store _ t@(TensorWrap x) : st6 -> do
          when (not $ prop_no_read_tensor_after_write_var st4 x (_vid v)) $
            throwError $ Continue (st1 ++ [head st2]) (tail st2)
          return $ st1 ++ [Bind v t] ++ st4 ++ st6

opt_remove_synonym :: Element a => PipelineStep a
opt_remove_synonym st = do
  -- liftIO $ putStrLn "opt_remove_synonym"
  let (st1, st2) = break isBind st
  when (null st2) $ throwError NotEligible
  case st2 of
    Bind v t : st3 -> do
      let (st4, st5) = break (isBindToTensor t) st3
      when (null st5) $ throwError $ Continue (st1 ++ [head st2]) (tail st2)
      case st5 of
        Bind v' _ : st6 -> do
          case cast v' of
            Nothing -> error "Binding a tensor with variables of different dimensions!"
            Just v' -> return $ st1 ++ [Bind v t] ++ st4 ++ substitute v' v st6

opt_absorb_gemv_dotadd :: Element a => PipelineStep a
opt_absorb_gemv_dotadd st = do
  -- liftIO $ putStrLn "opt_absorb_gemv_dotadd"
  lookfor st isGEMV (\(BlasGEMV _ _ _ _ _ _ v3) -> isDotAddTo $ _vid_VarWithDim v3) $
    \st1 g@(BlasGEMV o t v1 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotAdd v4 v5 v6 : st6 -> do
          if (b == 0) then
            if _vid v4 == _vid_VarWithDim v3 then
              return $ st1 ++ st4 ++ copy v5 v6 ++ [BlasGEMV o t v1 v2 a 1 (replace v3 v6)] ++ st6
            else
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 a 1 (replace v3 v6)] ++ st6
          else
            if _vid v4 == _vid v5 then
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 (2*a) (2*b) (replace v3 v6)] ++ st6
            else
              throwError $ Continue (st1 ++ [g]) (st4++st5)

opt_absorb_gemv_dotsca :: Element a => PipelineStep a
opt_absorb_gemv_dotsca st = do
  -- liftIO $ putStrLn "opt_absorb_gemv_dotsca"
  lookfor st isGEMV (\(BlasGEMV _ _ _ _ _ _ v3) -> isDotScaTo $ _vid_VarWithDim v3) $
    \st1 g@(BlasGEMV o t v1 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotSca c v4 v6 : st6 ->
          case cast c of
            Nothing -> error "This should not happen: types do not agree"
            Just c  -> return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 (c*a) (c*b) (replace v3 v6)] ++ st6

opt_absorb_gemm_dotadd :: Element a => PipelineStep a
opt_absorb_gemm_dotadd st = do
  -- liftIO $ putStrLn "opt_absorb_gemm_dotadd"
  lookfor st isGEMM (\(BlasGEMM _ _ _ _ _ _ _ v3) -> isDotAddTo $ _vid_VarWithDim v3) $
    \st1 g@(BlasGEMM o t1 v1 t2 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotAdd v4 v5 v6 : st6 -> do
          if (b == 0) then
            if _vid v4 == _vid_VarWithDim v3 then
              return $ st1 ++ st4 ++ copy v5 v6 ++ [BlasGEMM o t1 v1 t2 v2 a 1 (replace v3 v6)] ++ st6
            else
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 a 1 (replace v3 v6)] ++ st6
          else
            if _vid v4 == _vid v5 then
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 (2*a) (2*b) (replace v3 v6)] ++ st6
            else
              throwError $ Continue (st1 ++ [g]) (st4++st5)

opt_absorb_gemm_dotsca :: Element a => PipelineStep a
opt_absorb_gemm_dotsca st = do
  -- liftIO $ putStrLn "opt_absorb_gemm_dotsca"
  lookfor st isGEMM (\(BlasGEMM _ _ _ _ _ _ _ v3) -> isDotScaTo $ _vid_VarWithDim v3) $
    \st1 g@(BlasGEMM o t1 v1 t2 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotSca c v4 v6 : st6 ->
          case cast c of
            Nothing -> error "This should not happen: types do not agree"
            Just c  -> return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 (c*a) (c*b) (replace v3 v6)] ++ st6

lookfor st beg end act = do
  let (st1, st2) = break beg st
  when (null st2) $ throwError NotEligible
  case st2 of
    g : st3 -> do
      let (st4, st5) = break (end g) st3
      -- st == st1 ++ [g] ++ st4 ++ st5
      -- g matches beg
      -- head of st5 matches end if exists
      act st1 g st4 st5

copy v1 v2 = if v1 == v2 then [] else [Copy v1 v2]
