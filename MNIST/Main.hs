{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances, ScopedTypeVariables #-}
module Main where

import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.HMatrix
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import Parser

import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
import Control.Monad
import Control.Monad.Except

import Data.Constraint
foo :: Backend b s => b -> s -> Inp (ConvertFromSpec s) -> Env b (Out (ConvertFromSpec s))
foo b s i = case witness b s of Dict -> compile b s >>= run . flip forward i

main = do x <- runExceptT body
          case x of
            Left  _ -> putStrLn "Error."
            Right _ -> putStrLn "Done."

  where
    body = do
      cnn <- compile ByHmatrix (In2D 28 28 :++ Reshape2DAs1D :++ FullConnect 128 :++ FullConnect 10)
      liftIO $ putStrLn "Load training data."
      dataset <- liftIO $ uncurry zip <$> trainingData
      liftIO $ putStrLn "Load test data."
      liftIO $ putStrLn "Learning."
      cnn <- iterateM 15 (online dataset) cnn
      dotest cnn

dotest :: (Component n, Inp n ~ Image, Out n ~ Label, MonadIO m, RunInEnv (Run n) m)
       => n -> m ()
dotest !nn = do
    testset <- liftIO $ (uncurry zip <$> testData)
    liftIO $ putStrLn "Start test"
    result  <- mapM ((postprocess <$>) . run . forward nn . fst) testset
    let expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    liftIO $ putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

online :: (Component n, Inp n ~ Image, Out n ~ Label, MonadIO m, RunInEnv (Run n) m)
       => [(Inp n, Out n)] -> n -> m n
online ds !nn = walk ds nn
  where
    walk []     !nn = return nn
    walk (d:ds) !nn = do !nn <- run $ learn outcost' 0.0010 nn d
                         walk ds nn
    outcost' a b = return $ zipVectorWith cost' a b

iterateM :: (MonadIO m) => Int -> (a -> m a) -> a -> m a
iterateM n f x = walk 0 x
  where
    walk !i !a | i == n    = return a
               | otherwise = do -- when (i `mod` 10 == 0) $ putStrLn ("Iteration " ++ show i)
                                liftIO $ putStrLn ("Iteration " ++ show i)
                                a <- f a
                                walk (i+1) a

postprocess :: Vector Float -> Int
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)
