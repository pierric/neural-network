{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
module Main where

import qualified Data.Vector as BV
import Control.Monad.Except
import Data.Data
import System.IO (hFlush, stdout)
import Data.IORef
import Data.List (partition)
import Text.Printf (printf)
import Text.PrettyPrint.Free hiding ((</>))
import Data.NeuralNetwork hiding (cost')
import Data.NeuralNetwork.Common
import Data.NeuralNetwork.Adapter
import Data.NeuralNetwork.Backend.BLASHS
import Corpus

main = do putStrLn "Start."
          (nv, trdata, tsdata) <- corpus 1000
          let cc = InStream 1 :++ Embedding nv :++ -- Debug "i0" :++
                   Flow (LSTM 128) :++ -- Debug "i1" :++
                   Cutoff 80 :++ Concat :++
                   FullConnect 400 :++ FullConnect 10
                    --  :: SpecInString :++ SpecFlow SpecLSTM :++ SpecCutoff :++ SpecConcat :++ SpecFullConnect :++ SpecFullConnect
              mc = compile ByBLASHS cc
                    --  :: Env ByBLASHS
                    --       (Stack
                    --         (Stream (Stack LSTM (RunLayer (T SinglVec)) (CL LSTM_Env_Transformer)))
                    --         (Stack Cutoff
                    --           (Stack Concat
                    --             (Stack
                    --               (Stack (RunLayer F) (RunLayer (T SinglVec)) CE)
                    --               (Stack (RunLayer F) (RunLayer (T SinglVec)) CE) CE) CE) CE) CE)
          x <- runExceptT $ mc
          case x of
            Left _   -> putStrLn "Error."
            Right nn -> do
              putStrLn "Loaded."
              loop nn (BV.take 200 trdata) (BV.take 20 trdata) 1
              -- o <- forward nn i
              -- putStrLn $ showPretty $ text "#" <+> prettyDenseVectorFloat o
              -- nn <- learn diff rate nn tr0
  where
     loop nn trd tsd cnt = do
       nn <- dotrain nn trd cnt
       dotest nn tsd
       putStr "Continue? (number):"
       hFlush stdout
       str <- getLine
       let next = (reads :: ReadS Int) str
       when (not $ null next) (loop nn trd tsd (fst $ head next))


dotrain :: (Component n, Inp n ~ Sentence, Out n ~ Sentiment, Run n ~ IO)
        => n -> BV.Vector (Inp n, Out n) -> Int -> IO n
dotrain nn dataset mcnt = do
  cnt <- newIORef 0 :: IO (IORef Int)
  let dispAndInc = do
        i <- readIORef cnt
        writeIORef cnt (i+1)
        putStrLn ("Iteration " ++ show i)
  iterateM mcnt nn ((dispAndInc >>) . online 0.0001 dataset)

dotest :: (Component n, Inp n ~ Sentence, Out n ~ Sentiment, Run n ~ IO)
       => n -> BV.Vector (Inp n, Out n) -> IO ()
dotest nn dataset = do
    putStrLn "Start test"
    result <- BV.mapM ((>>= postprocess) . forward nn . fst) dataset
    expect <- BV.mapM (postprocess . snd) dataset
    let (co,wr) = BV.partition (uncurry (==)) $ BV.zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (BV.length co) (BV.length wr)
    putStrLn $ "First 10 tests:"
    BV.forM_ (BV.take 10 dataset) $ \(ds,ev) -> do
      pv <- forward nn ds
      putStrLn $ showPretty $ text "+" <+> prettyDenseVectorFloat pv
      putStrLn $ showPretty $ text "*" <+> prettyDenseVectorFloat ev
  where
    postprocess :: Sentiment -> IO Int
    postprocess v = do
      a <- denseVectorToVector v
      return $ BV.maxIndex a

online :: (Component n, Inp n ~ Sentence, Out n ~ Sentiment, Run n ~ IO)
       => Float -> BV.Vector (Inp n, Out n) -> n -> IO n
online rate ds nn = walk (BV.toList ds) nn
  where
    walk []     nn = return nn
    walk (d:ds) nn = do nn <- learn outcost' rate nn d
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

data SpecCutoff = Cutoff Int deriving (Typeable, Data)
type Cutoff = Adapter IO [DenseVector Float] [DenseVector Float] Int

instance BodySize SpecCutoff where
  bsize (SV (D1 s)) (Cutoff n) = SF n (D1 s)

instance MonadError ErrCode m => TranslateBody m SpecCutoff where
  --
  type SpecToCom SpecCutoff = Cutoff
  trans (SV (D1 s)) (Cutoff n) = return $ cutoff n
    where
      cutoff :: Num a => Int -> Cutoff
      cutoff n = Adapter to back
      to inp = do
        let r = length inp
        z <- replicateM (n-r) (newDenseVector s)
        return (r, take n $ inp ++ z)
      back r odelta = do
        z <- replicateM (r-n) (newDenseVector s)
        return $ take r (odelta ++ z)
  trans _ _ = throwError ErrMismatch

data SpecConcat = Concat deriving (Typeable, Data)
type Concat = Adapter IO [DenseVector Float] (DenseVector Float) (Int, Int)

instance BodySize SpecConcat where
  bsize (SF m (D1 n)) Concat = D1 (m*n)

instance MonadError ErrCode m => TranslateBody m SpecConcat where
  type SpecToCom SpecConcat = Concat
  trans (SF m (D1 n)) Concat = return nconcat
    where
      nconcat :: Concat
      nconcat = Adapter to back
      to inp = do let vinp = BV.fromList inp
                  cv <- denseVectorConcat vinp
                  return ((BV.length vinp, size (BV.head vinp)), cv)
      back (m,n) odelta = do let videlta = denseVectorSplit m n odelta
                             return $ BV.toList videlta
  trans _ _ = throwError ErrMismatch

data SpecEmbedding = Embedding Int deriving (Typeable, Data)
type Embedding = Adapter IO [Int] [DenseVector Float] ()

instance BodySize SpecEmbedding where
  bsize (SV (D1 1)) (Embedding n)= SV (D1 n)

instance MonadError ErrCode m => TranslateBody m SpecEmbedding where
  type SpecToCom SpecEmbedding = Embedding
  trans (SV (D1 s)) (Embedding n) = return $ Adapter to back
    where
      to inp = do v <- mapM (newDenseVectorDelta n) inp
                  return ((), v)
      back () odelta = return $ repeat 0
      newDenseVectorDelta n i = do
        v <- newDenseVector n
        unsafeWriteV v i 1
        return v

instance Pretty (DenseVector Float) where
  pretty = prettyDenseVectorFloat
showPretty x = displayS (renderPretty 0.4 500 x) ""
