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
import Control.Monad.IO.Class
import Control.Monad.Except
import System.Environment
import Text.PrettyPrint hiding (flatten)
import Text.PrettyPrint.HughesPJClass
import System.IO (hFlush, stdout)
import System.IO.Unsafe
import Parser

main = do x <- runExceptT $ compile byBLASHSf (In2D 28 28,
                                              Convolution 2 7 3 :&: MaxPooling 2 :&:
                                              Convolution 4 5 2 :&: MaxPooling 2 :&:
                                              Reshape2DAs1D :&:
                                              FullConnect 512 :&: FullConnect 32 :&:
                                              FullConnect 10 :&: HNil,
                                              MeanSquaredError)
          case x of
            Left _ -> putStrLn "Error."
            Right cnn -> do
              loop cnn 5
              -- debug cnn
  where
    loop cnn cnt = do
      cnn <- dotrain cnn cnt
      dotest cnn
      putStr "Continue? (number):"
      hFlush stdout
      str <- getLine
      let next = (reads :: ReadS Int) str
      when (not $ null next) (loop cnn (fst $ head next))

debug :: (ModelCst n e, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
      => (n,e) -> IO ()
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
    smalltest it (nn,_) = do
      flip mapM_ it $ \(ds,ev) -> do
        pv <- forward nn ds
        prettyResult pv >>= putStrLn . ("+" ++ )
        prettyResult ev >>= putStrLn . ("*" ++ )

dotrain :: (ModelCst n e, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
        => (n,e)-> Int -> IO (n,e)
dotrain nn mcnt = do
  putStrLn "Load training data."
  dataset <- trainingData >>= mapM preprocess . uncurry zip
  putStrLn "Load test data."
  putStrLn "Learning."
  cnt <- newIORef 0 :: IO (IORef Int)
  let dispAndInc = do
        i <- readIORef cnt
        writeIORef cnt (i+1)
        putStrLn ("Iteration " ++ show i)
  iterateM mcnt nn ((dispAndInc >>) . online 0.001 dataset)

dotest :: (ModelCst n e, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
       => (n,e) -> IO ()
dotest !(nn,_) = do
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

online :: (ModelCst n e, Inp n ~ PImage, Out n ~ PLabel, Run n ~ IO)
       => Float -> [(Inp n, Out n)] -> (n,e) -> IO (n,e)
online rate ds !nn = walk ds nn
  where
    walk []     !nn = return nn
    walk (d:ds) !nn = do !nn <- learn nn d rate
                         walk ds nn

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
  a <- denseVectorToVector v
  return $ V.maxIndex a

prettyResult a = do
    v <- postprocess a
    return $ showPretty $ text (printf "%02d:" v) <+> pPrint a
  where
    showPretty = renderStyle (Style PageMode 500 0.4)
