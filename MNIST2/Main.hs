{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances, ScopedTypeVariables #-}
module Main where

import Data.NeuronNetwork
import Data.NeuronNetwork.Backend.BLASHS
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
import Control.Monad
import Control.Monad.Except
import System.Environment
import Text.PrettyPrint.Free hiding (flatten)
import System.IO.Unsafe
import Parser

main = do x <- runExceptT $ compile ByBLASHS (In1D 768 :++ FullConnect 256 :++ FullConnect 10)
          case x of
            Left _ -> putStrLn "Error."
            Right cnn -> debug cnn -- dotrain cnn >>= dotest

debug :: (Component n, Inp n ~ DenseVector Float, Out n ~ DenseVector Float, Run n ~ IO)
      => n -> IO ()
debug nn = do
  a0:a1:_ <- getArgs
  let cycle = read a0 :: Int
      rate  = read a1 :: Float
  putStrLn "Load training data."
  dataset <- trainingData >>= mapM preprocess . uncurry zip
  nn <- iterateM cycle (online rate dataset) nn
  putStrLn "Load test data."
  testset <- testData >>= mapM preprocess . uncurry zip
  flip mapM_ (take 10 testset) $ \(ds,ev) -> do
    pv <- forward nn ds
    prettyResult pv >>= putStrLn . ("+" ++ )
    prettyResult ev >>= putStrLn . ("*" ++ )

dotrain :: (Component n, Inp n ~ DenseVector Float, Out n ~ DenseVector Float, Run n ~ IO)
        => n -> IO n
dotrain nn = do
  putStrLn "Load training data."
  dataset <- trainingData >>= mapM preprocess . uncurry zip
  putStrLn "Load test data."
  putStrLn "Learning."
  iterateM 15 (online 0.0010 dataset) nn

dotest :: (Component n, Inp n ~ DenseVector Float, Out n ~ DenseVector Float, Run n ~ IO)
       => n -> IO ()
dotest !nn = do
    testset <- testData >>= mapM preprocess . uncurry zip
    putStrLn "Start test"
    result <- mapM ((>>= postprocess) . forward nn . fst) testset
    expect <- mapM (postprocess . snd) testset
    let (co,wr) = partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

online :: (Component n, Inp n ~ DenseVector Float, Out n ~ DenseVector Float, Run n ~ IO)
       => Float -> [(Inp n, Out n)] -> n -> IO n
online rate ds !nn = walk ds nn
  where
    walk []     !nn = return nn
    walk (d:ds) !nn = do !nn <- learn outcost' rate nn d
                         walk ds nn
    outcost' a b = do v <- newDenseVector (size a)
                      v <<= ZipWith cost' a b
                      return v

iterateM :: (MonadIO m) => Int -> (a -> m a) -> a -> m a
iterateM n f x = walk 0 x
  where
    walk !i !a | i == n    = return a
               | otherwise = do -- when (i `mod` 10 == 0) $ putStrLn ("Iteration " ++ show i)
                                liftIO $ putStrLn ("Iteration " ++ show i)
                                a <- f a
                                walk (i+1) a

preprocess :: (Image, Label) -> IO (DenseVector Float, DenseVector Float)
preprocess (img,lbl) = do
  i <- SV.unsafeThaw img
  l <- SV.unsafeThaw lbl
  return (DenseVector i, DenseVector l)
postprocess :: DenseVector Float -> IO Int
postprocess v = do
  a <- toListV v
  return $ fst $ maximumBy cmp $ zip [0..] a
  where cmp a b = compare (snd a) (snd b)

prettyResult a = do
    v <- postprocess a
    return $ showPretty $ text (printf "%02d:" v) <+> pretty a
  where
    showPretty x = displayS (renderPretty 0.4 500 x) ""

instance Pretty (DenseVector R) where
  pretty vec = let a = unsafePerformIO (toListV vec)
               in encloseSep langle rangle comma $ map (text . printf "%.04f") a
