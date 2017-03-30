module Optimize where

import Tensor
import Prop
import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Identity
import Data.Typeable (cast)

-- import Debug.Trace
-- import Text.PrettyPrint.Free (pretty)

data Error = NotEligible | Continue [Statement] [Statement]

opt_rewrite_alloc_store_as_bind :: [Statement] -> [Statement]
opt_rewrite_alloc_store_as_bind st =
  case runIdentity $ runExceptT act of
    Left NotEligible        -> st
    Left (Continue st1 st2) -> st1 ++ opt_rewrite_alloc_store_as_bind st2
    Right st1               -> st1
  where
    act = do let (st1, st2) = break isAlloc st
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


test :: IO [Statement]
test = do
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
