{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances, ScopedTypeVariables #-}
module Main where

import Data.NeuralNetwork hiding (cost')
import Data.NeuralNetwork.Backend.BLASHS
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Data.List(foldl',partition,maximumBy)
import Data.IORef
import Text.Printf (printf)
import Control.Monad
import Control.Monad.Except
import System.Environment
import Text.PrettyPrint.Free hiding (flatten)
import System.IO.Unsafe
import Parser

main = do x <- runExceptT $ compile ByBLASHS (In2D 28 28 :++
                                              Convolution 4 7 3 :++ MaxPooling 2 :++
                                              Reshape2DAs1D :++
                                              FullConnect 256 :++ FullConnect 64 :++
                                              FullConnect 10)
          case x of
            Left _ -> putStrLn "Error."
            Right cnn -> -- dotrain cnn >>= dotest
                         debug cnn

debug :: (Component n, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
      => n -> IO ()
debug nn = do
  a0:a1:_ <- getArgs
  let cycle = read a0 :: Int
      rate  = read a1 :: Float
  putStrLn "Load training data."
  dataset <- trainingData >>= mapM preprocess . uncurry zip
  testset <- testData >>= mapM preprocess . take 10 . uncurry zip
  cnt <- newIORef 0 :: IO (IORef Int)
  let dispAndInc = do
        i <- readIORef cnt
        writeIORef cnt (i+1)
        putStrLn ("Iteration " ++ show i)
  nn <- iterateM (cycle `div` checkpoint) nn $ \nn1 -> do
          nn1 <- iterateM checkpoint nn1 $ (dispAndInc >>) . online rate dataset
          putStrLn "[test]..."
          smalltest testset nn1
          return nn1
  nn <- iterateM (cycle `mod` checkpoint) nn $ (dispAndInc >>) . online rate dataset
  putStrLn "[final test]..."
  smalltest testset nn
  where
    checkpoint = 2
    smalltest it nn = do
      flip mapM_ it $ \(ds,ev) -> do
        pv <- forward nn ds
        prettyResult pv >>= putStrLn . ("+" ++ )
        prettyResult ev >>= putStrLn . ("*" ++ )

dotrain :: (Component n, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
        => n -> IO n
dotrain nn = do
  putStrLn "Load training data."
  dataset <- trainingData >>= mapM preprocess . uncurry zip
  putStrLn "Load test data."
  putStrLn "Learning."
  cnt <- newIORef 0 :: IO (IORef Int)
  let dispAndInc = do
        i <- readIORef cnt
        writeIORef cnt (i+1)
        putStrLn ("Iteration " ++ show i)
  iterateM 15 nn ((dispAndInc >>) . online 0.002 dataset)

dotest :: (Component n, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
       => n -> IO ()
dotest !nn = do
    testset <- testData >>= mapM preprocess . uncurry zip
    putStrLn "Start test"
    result <- mapM ((>>= postprocess) . forward nn . fst) testset
    expect <- mapM (postprocess . snd) testset
    let (co,wr) = partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)
    putStrLn $ "First 10 tests:"
    flip mapM_ (take 10 testset) $ \(ds,ev) -> do
      pv <- forward nn ds
      prettyResult pv >>= putStrLn . ("+" ++ )
      prettyResult ev >>= putStrLn . ("*" ++ )

online :: (Component n, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
       => Float -> [(Inp n, Out n)] -> n -> IO n
online rate ds !nn = walk ds nn
  where
    walk []     !nn = return nn
    walk (d:ds) !nn = do !nn <- learn outcost' rate nn d
                         walk ds nn
    outcost' a b = do v <- newDenseVector (size a)
                      v <<= ZipWith cost' a b
                      return v

iterateM :: (MonadIO m) => Int -> a -> (a -> m a) -> m a
iterateM n x f = go 0 x
  where
    go i x = if i == n
             then
               return x
             else do
               x <- f x
               go (i+1) x

type PImage = V.Vector (DenseMatrix Float)
type PLabel = DenseVector Float
preprocess :: (Image, Label) -> IO (PImage, PLabel)
preprocess (img,lbl) = do
  i <- SV.unsafeThaw img
  l <- SV.unsafeThaw lbl
  return (V.singleton $ DenseMatrix 28 28 i, DenseVector l)
postprocess :: PLabel -> IO Int
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
