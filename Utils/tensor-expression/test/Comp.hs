{-# LANGUAGE FlexibleInstances, FlexibleContexts, ScopedTypeVariables #-}
module Comp where

import Test.QuickCheck
import qualified Data.Vector.Storable as PV
import Data.Maybe
import Data.Typeable (cast)
import Control.Monad
import Control.Monad.State
import Control.Monad.Writer hiding ((<>))
import Data.Tensor
import qualified Data.Tensor.Compile as C
import Gen

diff_ce :: Element a => C.ExprHashed a -> C.ExprHashed a -> [(C.ExprHashed a, C.ExprHashed a)]
diff_ce e1 e2 = execWriter (go e1 e2)
  where
    go e1 e2 | C.L v1 e1s1 e1s2 <- C.body e1
             , C.L v2 e2s1 e2s2 <- C.body e2
             , v1 == v2             = do go e1s1 e2s1
                                         go e1s2 e2s2

             | C.V v1 <- C.body e1
             , C.V v2 <- C.body e2
             , v1 == v2             = return ()

             | C.I t1 <- C.body e1
             , C.I t2 <- C.body e2
             , t1 == t2             = return ()

             | C.S f1 s1 <- C.body e1
             , C.S f2 s2 <- C.body e2
             , f1 == f2             = go s1 s2

             | C.Bin o1 e1s1 e1s2 <- C.body e1
             , C.Bin o2 e2s1 e2s2 <- C.body e2
             , o1 == o2             = do go e1s1 e2s1
                                         go e1s2 e2s2

             | otherwise = tell [(e1,e2)]

insert_ce :: (Element a, Arbitrary a) => Int -> C.ExprHashed a -> C.ExprHashed a -> IO (C.ExprHashed a, Int)
insert_ce n s e = runStateT (go replace e) 0
  where
    go :: Monad m => (C.ExprAttr a e -> m (C.ExprAttr a e)) -> C.ExprAttr a e -> m (C.ExprAttr a e)
    go f (a C.:@ C.L v x y)   = do x <- go f x
                                   y <- go f y
                                   return $ a C.:@ C.L v x y
    go f (a C.:@ C.S u x)     = do x <- go f x
                                   return $ a C.:@ C.S u x
    go f (a C.:@ C.Bin o x y) = do x <- go f x
                                   y <- go f y
                                   return $ a C.:@ C.Bin o x y
    go f e                    = f e

    count e = modify (+1) >> return e
    leaf = execState (go count e) 0

    replace e = do g <- liftIO $ generate $ choose (0,leaf)
                   if g <= n 
                     then do
                       modify (+1)
                       liftIO $ generate $ adapt e s
                     else 
                       return e
    adapt e s = case (C.dim_ce e, C.dim_ce s) of
                  (C.DimWrap d1, C.DimWrap d2)
                    | Just (D1 a)   <- cast d1
                    , Just (D1 b)   <- cast d2 -> do x <- gen2 (D2 a b)
                                                     return $ C.mk (C.dim_ce e) $ C.Bin C.MV x s
                    | Just (D1 a)   <- cast d1
                    , Just (D2 b c) <- cast d2 -> do x <- gen1 (D1 b)
                                                     y <- gen2 (D2 c a)
                                                     z <- return $ C.mk (C.DimWrap (D1 c)) $ C.Bin C.VM x s
                                                     return $ C.mk (C.dim_ce e) $ C.Bin C.VM z y
                    | Just (D2 a b) <- cast d1
                    , Just (D1 c)   <- cast d2 -> do x <- gen2 (D2 a c)
                                                     y <- gen1 (D1 b)
                                                     z <- return $ C.mk (C.DimWrap (D1 a)) $ C.Bin C.MV x s
                                                     return $ C.mk (C.dim_ce e) $ C.Bin C.OVV z y
                    | Just (D2 a b) <- cast d1
                    , Just (D2 c d) <- cast d2 -> do x <- gen2 (D2 b d)
                                                     y <- gen2 (D2 a c)
                                                     z <- return $ C.mk (C.DimWrap (D2 b c)) $ C.Bin C.MTM s x
                                                     return $ C.mk (C.dim_ce e) $ C.Bin C.MTM z y
      where
        gen2 d = liftM (C.compile . fromJust) $ runGenM 2 (genExp2D d)
        gen1 d = liftM (C.compile . fromJust) $ runGenM 2 (genExp1D d)

notVI :: C.ExprHashed Float -> Bool
notVI (_ C.:@ C.I _)   = False
notVI (_ C.:@ C.V _)   = False
notVI (_ C.:@ C.S _ e) = notVI e
notVI _                = True