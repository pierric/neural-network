module Data.Tensor.Prop where

import Data.Tensor.Class
import Data.Tensor.Execute
import qualified Data.Set as S
import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Identity
import Data.Either (isRight)
import Data.Maybe (catMaybes)
import Data.List (nub, elem)
import Data.Monoid (Any(..), mconcat)

data MakeType = MakeAlloc | MakeBind
  deriving (Eq, Ord, Show)
data VarUsage = Make  { _vu_mt :: MakeType, _vuid :: Int }
              | Read  { _vuid :: Int }
              | Write { _vuid :: Int }
  deriving (Eq, Ord, Show)

isMake (Make _ _) = True
isMake _          = False

vars :: [Statement] -> [VarUsage]
vars = nub . concat . map varUsage

varUsage :: Statement -> [VarUsage]
varUsage (Alloc v)   = [Make MakeAlloc $ _vid v]
varUsage (Bind  v _) = [Make MakeBind  $ _vid v]
varUsage (Store v _) = [Write $ _vid v]
varUsage (Copy  v w) = [Read (_vid v), Write (_vid w)]
varUsage (BlasGEMV _ _ v1 v2 _ _ v3)   = [Read $ _vid v1,Read $ _vid v2,Write $ _vid v3]
varUsage (BlasGERU _ v1 v2 _ v3)       = [Read $ _vid v1,Read $ _vid v2,Write $ _vid v3]
varUsage (BlasGEMM _ _ v1 _ v2 _ _ v3) = [Read $ _vid v1,Read $ _vid v2,Write $ _vid v3]
varUsage (DotAdd v1 v2 v3)             = [Read $ _vid v1,Read $ _vid v2,Write $ _vid v3]
varUsage (DotMul v1 v2 v3)             = [Read $ _vid v1,Read $ _vid v2,Write $ _vid v3]
varUsage (DotSca _ v1 v2)              = [Read $ _vid v1,Write $ _vid v2]

data Error = MakeTwice     VarUsage
           | UseBeforeMake VarUsage
  deriving Show

read_of_var  vid s = Read  vid `elem` varUsage s
write_of_var vid s = Write vid `elem` varUsage s
bind_of_tensor :: Dimension d => Tensor d a -> [Statement] -> [Int]
bind_of_tensor t st = catMaybes $ map match st
  where
    match (Bind v t1) = if eqTensor t t1 then Just (_vid v) else Nothing
    match _           = Nothing

prop_no_make_twice st = isRight $ runIdentity $ runExceptT $ runStateT (check st) S.empty
  where
    check [] = return ()
    check (s:ss) = do
      m <- get
      forM (filter isMake $ varUsage s) $ \v ->
        if S.member (_vuid v) m then
          throwError (MakeTwice v)
        else
          modify (S.insert $ _vuid $ v)
      check ss

prop_no_use_before_make st = isRight $ runIdentity $ runExceptT $ runStateT (check st) S.empty
  where
    check [] = return ()
    check (s:ss) = do
      m <- get
      forM (varUsage s) $ \v ->
        if isMake v then
          modify (S.insert $ _vuid v)
        else if S.member (_vuid v) m then
          return ()
        else
          throwError (UseBeforeMake v)
      check ss

-- prop_no_read_tensor_after_write_var :: Dimension d => [Statement] -> Tensor d a -> Int -> Bool
prop_no_read_tensor_after_write_var st t vid =
  let ist = zip [1::Int ..] st
      rt  = map fst $ filter (getAny . (mconcat $ map ((Any .) . read_of_var) $ bind_of_tensor t st) . snd) ist
      wv  = map fst $ filter (write_of_var vid . snd) ist
  in null wv || null rt || minimum wv > maximum rt

prop_only_make_var :: [Statement] -> Bool
prop_only_make_var = and . map isMake . concatMap varUsage
