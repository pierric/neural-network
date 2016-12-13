{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances, ScopedTypeVariables #-}
module Main where

import Data.NeuralNetwork
import Data.NeuralNetwork.Backend.HMatrix
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import Parser

import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
import Control.Monad
import Control.Monad.Except
import Text.PrettyPrint.Free hiding (flatten, (<>))
import System.Environment

import Data.Constraint
foo :: Backend b s => b -> s -> Inp (ConvertFromSpec s) -> Env b (Out (ConvertFromSpec s))
foo b s i = case witness b s of Dict -> compile b s >>= run . flip forward i

main = do x <- runExceptT $ do
                cnn <- compile ByHmatrix (In2D 28 28 :++ Reshape2DAs1D :++ FullConnect 256 :++ FullConnect 32 :++ FullConnect 10)
                debug cnn -- dotrain cnn >>= dotest
          case x of
            Left  _ -> putStrLn "Error."
            Right _ -> putStrLn "Done."

debug :: (Component n, Inp n ~ Image, Out n ~ Label, MonadIO m, RunInEnv (Run n) m)
      => n -> m ()
debug nn = do
    a0:a1:_ <- liftIO $ getArgs
    let cycle = read a0 :: Int
        rate  = read a1 :: Float
    liftIO $ putStrLn "Load training data."
    dataset <- liftIO $ uncurry zip <$> trainingData
    nn <- iterateM cycle (online rate dataset) nn
    liftIO $ putStrLn "Load test data."
    testset <- liftIO $ (uncurry zip <$> testData)
    flip mapM_ (take 10 testset) $ \(ds,ev) -> do
      pv <- run $ forward nn ds
      liftIO $ putStrLn ("+" ++ showPretty (text (printf "%02d:" $ postprocess pv) <+> pretty pv))
      liftIO $ putStrLn ("*" ++ showPretty (text (printf "%02d:" $ postprocess ev) <+> pretty ev))

dotrain :: (Component n, Inp n ~ Image, Out n ~ Label, MonadIO m, RunInEnv (Run n) m)
       => n -> m n
dotrain nn = do
    liftIO $ putStrLn "Load training data."
    dataset <- liftIO $ uncurry zip <$> trainingData
    liftIO $ putStrLn "Learning."
    iterateM 5 (online 0.0001 dataset) nn

dotest :: (Component n, Inp n ~ Image, Out n ~ Label, MonadIO m, RunInEnv (Run n) m)
       => n -> m ()
dotest !nn = do
    liftIO $ putStrLn "Load test data."
    testset <- liftIO $ (uncurry zip <$> testData)
    liftIO $ putStrLn "Start test"
    result  <- mapM ((postprocess <$>) . run . forward nn . fst) testset
    let expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    liftIO $ putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

online :: (Component n, Inp n ~ Image, Out n ~ Label, MonadIO m, RunInEnv (Run n) m)
       => Float -> [(Inp n, Out n)] -> n -> m n
online rate ds !nn = walk ds nn
  where
    walk []     !nn = return nn
    walk (d:ds) !nn = do !nn <- run $ learn outcost' rate nn d
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

instance Pretty (Vector Float) where
  pretty vec = encloseSep langle rangle comma $ map (text . printf "%.04f") $ toList vec

showPretty x = displayS (renderPretty 0.4 500 x) ""
