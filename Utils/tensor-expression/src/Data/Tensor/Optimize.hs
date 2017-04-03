module Optimize where

import Tensor
import Prop
import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Identity
import Data.Typeable (cast)

-- import Debug.Trace
-- import Text.PrettyPrint.Free (pretty)

data OptError = NotEligible | Continue [Statement] [Statement]
type OptMR = StateT CGState IO
type OptimizerR = [Statement] -> OptMR [Statement]
type OptMP = ExceptT OptError OptMR
type OptimizerP = [Statement] -> OptMP [Statement]

run_opt :: OptimizerP -> OptimizerR
run_opt act st = do
  r <- runExceptT (act st)
  case r of
    Left NotEligible        -> return $ st
    Left (Continue st1 st2) -> run_opt act st2 >>= return . (st1 ++)
    Right st1               -> return $ st1

opt_rewrite_alloc_store_as_bind :: OptimizerP
opt_rewrite_alloc_store_as_bind st = do
  let (st1, st2) = break isAlloc st
  when (null st2) $ throwError NotEligible
  case st2 of
    Alloc v : st3 -> do
      let (st4, st5) = break (isStoreTo $ _vid v) st3
      when (null st5) $ throwError $ Continue (st1 ++ [head st2]) (tail st2)
      case st5 of
        Store _ x : st6 -> do
          when (not $ prop_no_read_tensor_after_write_var st4 x (_vid v)) $
            throwError $ Continue (st1 ++ [head st2]) (tail st2)
          case cast x of
            Just x' -> return $ st1 ++ [Bind v x'] ++ st4 ++ st6

opt_remove_synonym :: OptimizerP
opt_remove_synonym st = do
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

opt_absorb_gemv_dotadd :: OptimizerP
opt_absorb_gemv_dotadd st =
  lookfor st isGEMV (\(BlasGEMV _ _ _ _ _ _ v3) -> isDotScaTo $ _vid v3) $
    \st1 g@(BlasGEMV o t v1 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotAdd v4 v5 v6 : st6 -> do
          if (b == 0) then
            if _vid v4 == _vid v3 then
              return $ st1 ++ st4 ++ copy v5 v6 ++ [BlasGEMV o t v1 v2 a 1 (v3{_vid = _vid v6})] ++ st6
            else
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 a 1 (v3{_vid = _vid v6})] ++ st6
          else
            if _vid v4 == _vid v5 then
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 (2*a) (2*b) (v3{_vid = _vid v6})] ++ st6
            else
              throwError $ Continue (st1 ++ [g]) (st4++st5)

  -- let (st1, st2) = break isGEMV st
  -- when (null st2) $ throwError NotEligible
  -- case st2 of
  --   g@(BlasGEMV o t v1 v2 a b v3) : st3 -> do
  --     let (st4, st5) = break (isDotAddTo $ _vid v3) st3
  --     when (not (prop_only_make_var st4) || null st5) $
  --      throwError $ Continue (st1 ++ [head st2]) (tail st2)
  --     case st5 of
  --       DotAdd v4 v5 v6 : st6 -> do
  --         if (b == 0) then
  --           if _vid v4 == _vid v3 then
  --             return $ st1 ++ st4 ++ copy v5 v6 ++ [BlasGEMV o t v1 v2 a 1 (v3{_vid = _vid v6})] ++ st6
  --           else
  --             return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 a 1 (v3{_vid = _vid v6})] ++ st6
  --         else
  --           if _vid v4 == _vid v5 then
  --             return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 (2*a) (2*b) (v3{_vid = _vid v6})] ++ st6
  --           else
  --             throwError $ Continue (st1 ++ [head st2]) (tail st2)

opt_absorb_gemv_dotsca :: OptimizerP
opt_absorb_gemv_dotsca st =
  lookfor st isGEMV (\(BlasGEMV _ _ _ _ _ _ v3) -> isDotScaTo $ _vid v3) $
    \st1 g@(BlasGEMV o t v1 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotSca c v4 v6 : st6 ->
          case cast c of
            Nothing -> error "This should not happen: types do not agree"
            Just c  -> return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 (c*a) (c*b) (v3{_vid = _vid v6})] ++ st6

  -- let (st1, st2) = break isGEMV st
  -- when (null st2) $ throwError NotEligible
  -- case st2 of
  --   g@(BlasGEMV o t v1 v2 a b v3) : st3 -> do
  --     let (st4, st5) = break (isDotScaTo $ _vid v3) st3
  --     when (not (prop_only_make_var st4) || null st5) $
  --      throwError $ Continue (st1 ++ [head st2]) (tail st2)
  --     case st5 of
  --       DotSca c v4 v6 : st6 ->
  --         case cast c of
  --           Nothing -> error "This should not happen: types do not agree"
  --           Just c  -> return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMV o t v1 v2 (c*a) (c*b) (v3{_vid = _vid v6})] ++ st6

opt_absorb_gemm_dotadd :: OptimizerP
opt_absorb_gemm_dotadd st =
  lookfor st isGEMM (\(BlasGEMM _ _ _ _ _ _ _ v3) -> isDotAddTo $ _vid v3) $
    \st1 g@(BlasGEMM o t1 v1 t2 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotAdd v4 v5 v6 : st6 -> do
          if (b == 0) then
            if _vid v4 == _vid v3 then
              return $ st1 ++ st4 ++ copy v5 v6 ++ [BlasGEMM o t1 v1 t2 v2 a 1 (v3{_vid = _vid v6})] ++ st6
            else
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 a 1 (v3{_vid = _vid v6})] ++ st6
          else
            if _vid v4 == _vid v5 then
              return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 (2*a) (2*b) (v3{_vid = _vid v6})] ++ st6
            else
              throwError $ Continue (st1 ++ [g]) (st4++st5)

  -- let (st1, st2) = break isGEMM st
  -- when (null st2) $ throwError NotEligible
  -- case st2 of
  --   g@(BlasGEMM o t1 v1 t2 v2 a b v3) : st3 -> do
  --     let (st4, st5) = break (isDotAddTo $ _vid v3) st3
  --     when (not (prop_only_make_var st4) || null st5) $
  --      throwError $ Continue (st1 ++ [head st2]) (tail st2)
  --     case st5 of
  --       DotAdd v4 v5 v6 : st6 -> do
  --         if (b == 0) then
  --           if _vid v4 == _vid v3 then
  --             return $ st1 ++ st4 ++ copy v5 v6 ++ [BlasGEMM o t1 v1 t2 v2 a 1 (v3{_vid = _vid v6})] ++ st6
  --           else
  --             return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 a 1 (v3{_vid = _vid v6})] ++ st6
  --         else
  --           if _vid v4 == _vid v5 then
  --             return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 (2*a) (2*b) (v3{_vid = _vid v6})] ++ st6
  --           else
  --             throwError $ Continue (st1 ++ [head st2]) (tail st2)

opt_absorb_gemm_dotsca :: OptimizerP
opt_absorb_gemm_dotsca st =
  lookfor st isGEMV (\(BlasGEMM _ _ _ _ _ _ _ v3) -> isDotScaTo $ _vid v3) $
    \st1 g@(BlasGEMM o t1 v1 t2 v2 a b v3) st4 st5 -> do
      when (not (prop_only_make_var st4) || null st5) $
       throwError $ Continue (st1 ++ [g]) (st4++st5)
      case st5 of
        DotSca c v4 v6 : st6 ->
          case cast c of
            Nothing -> error "This should not happen: types do not agree"
            Just c  -> return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 (c*a) (c*b) (v3{_vid = _vid v6})] ++ st6

  -- let (st1, st2) = break isGEMV st
  -- when (null st2) $ throwError NotEligible
  -- case st2 of
  --   g@(BlasGEMM o t1 v1 t2 v2 a b v3) : st3 -> do
  --     let (st4, st5) = break (isDotScaTo $ _vid v3) st3
  --     when (not (prop_only_make_var st4) || null st5) $
  --      throwError $ Continue (st1 ++ [head st2]) (tail st2)
  --     case st5 of
  --       DotSca c v4 v6 : st6 ->
  --         case cast c of
  --           Nothing -> error "This should not happen: types do not agree"
  --           Just c  -> return $ st1 ++ st4 ++ copy v4 v6 ++ [BlasGEMM o t1 v1 t2 v2 (c*a) (c*b) (v3{_vid = _vid v6})] ++ st6

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

test1 :: IO [Statement]
test1 = do
  t1 <- newTensor (D1 4) :: IO (Tensor D1 Float)
  t2 <- newTensor (D2 4 4)
  t3 <- newTensor (D1 4)
  let cc = compile $ (I t1 :<# I t2) :.+ I t3
  er <- runExceptT $ evalStateT cc 0
  case er of
    Left  e -> error $ show e
    Right (st, vr) -> do
      t4 <- newTensor (D1 4)
      return $ st ++ [Store vr t4]

test2 :: IO [Statement]
test2 = do
  t1 <- newTensor (D1 4) :: IO (Tensor D1 Float)
  t2 <- newTensor (D1 4)
  let cc = compile $ (I t1 :.+ I t2)
  er <- runExceptT $ evalStateT cc 0
  case er of
    Left  e -> error $ show e
    Right (st, vr) -> do
      return $ st ++ [Store vr t1]
