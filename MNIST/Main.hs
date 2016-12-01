{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances, ScopedTypeVariables #-}
module Main where

import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.HMatrix
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import Parser

import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
import Control.Parallel.Strategies
import System.Environment
import Control.Monad
import Control.Monad.Except

import Data.Constraint
eval :: Backend b s => b -> s -> (ConvertFromSpec s -> ) -> Env b (Out (ConvertFromSpec s))
eval b s i = case witness b s of Dict -> compile b s >>= run . flip forward i

main = do x <- runExceptT body
          case x of
            Left  _ -> putStrLn "Error."
            Right _ -> putStrLn "Done."

  where
    body = do
      cnn <- compile ByHmatrix (In2D 28 28 :++ Convolution 1 7 3 :++ Reshape2DAs1D :++ FullConnect 128 :++ FullConnect 10)
      putStrLn "Load training data."
      dataset <- uncurry zip <$> trainingData
      putStrLn "Load test data."
      putStrLn "Learning."
      nn <- iterateM 15 (online dataset) nn
      dotest nn

dotest :: (Component n, Inp n ~ Image, Out n ~ Label) => n -> Env n ()
dotest !nn = do
    testset <- liftIO $ (uncurry zip <$> testData)
    liftIO $ putStrLn "Start test"
    result  <- mapM (postprocess . forward nn . fst) testset `using` parList rdeepseq
    let expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    liftIO $ putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

online ds !nn = walk ds nn
  where
    walk []     !nn = nn
    walk (d:ds) !nn = let !nn' = learn (zipVectorWith cost') 0.0010 nn d
                      in walk ds nn'

iterateM :: Int -> (a -> a) -> a -> IO a
iterateM n f x = walk 0 x
  where
    walk !i !a | i == n    = return a
               | otherwise = do -- when (i `mod` 10 == 0) $ putStrLn ("Iteration " ++ show i)
                                putStrLn ("Iteration " ++ show i)
                                walk (i+1) $! f a

postprocess :: Vector Float -> Int
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)
