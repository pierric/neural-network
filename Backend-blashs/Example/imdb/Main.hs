{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
module Main where

import qualified Data.Text as Text
import qualified Data.Text.Encoding as Text
import qualified Data.ByteString as BS
import qualified Data.Vector as BV
import qualified Data.Vector.Storable as SV
import Control.Monad.Except
import Data.Hashable
import Data.Data
import qualified NLP.Tokenize.Text as Token
import System.FilePath
import System.Directory
import System.IO (hFlush, stdout)
import Data.IORef
import Data.List (partition)
import Text.Printf (printf)
import Text.PrettyPrint.Free hiding ((</>))
import Data.NeuralNetwork hiding (cost')
import Data.NeuralNetwork.Common
import Data.NeuralNetwork.Adapter
import Data.NeuralNetwork.Backend.BLASHS

type Sentence = [DenseVector Float]
type Sentiment = DenseVector Float

load_imdb_train = do
  let path = "tdata" </> "aclImdb" </> "train" </> "pos"
  files <- take 1200 <$> listDirectory path
  tr_pos <- mapM (load_entry path) files
  let path = "tdata" </> "aclImdb" </> "train" </> "neg"
  files <- take 1200 <$> listDirectory path
  tr_neg <- mapM (load_entry path) files
  return $ tr_pos ++ tr_neg

load_imdb_test = do
  let path = "tdata" </> "aclImdb" </> "test" </> "pos"
  files <- take 10 <$> listDirectory path
  tr_pos <- mapM (load_entry path) files
  let path = "tdata" </> "aclImdb" </> "test" </> "neg"
  files <- take 10 <$> listDirectory path
  tr_neg <- mapM (load_entry path) files
  return $ tr_pos ++ tr_neg

load_entry :: FilePath -> FilePath -> IO (Sentence, Sentiment)
load_entry path file = do
  content <- BS.readFile (path </> file)
  let toks = Token.tokenize (Text.decodeUtf8 content)
      eval = let (_, '_':m) = break (=='_') file
                 (n, _) = break (=='.') m
             in read n - 1 :: Int
  toks' <- mapM (newDenseVectorConst 1 . (/16777216.0) . fromIntegral . hash) toks
  eval' <- DenseVector <$> (SV.unsafeThaw $ SV.fromList (replicate eval 0 ++ [1] ++ replicate (9-eval) 0))
  return (toks', eval')

main = do let cc = InStream :++ -- Debug "i0" :++
                   Flow (LSTM 1) :++ -- Debug "i1" :++
                   Cutoff 120 :++ Concat :++
                   FullConnect 200 :++ FullConnect 10
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
              putStrLn "OK."
              trdata <- load_imdb_train
              tsdata <- load_imdb_test
              loop nn trdata (take 10 trdata) 1
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
        => n -> [(Inp n, Out n)] -> Int -> IO n
dotrain nn dataset mcnt = do
  cnt <- newIORef 0 :: IO (IORef Int)
  let dispAndInc = do
        i <- readIORef cnt
        writeIORef cnt (i+1)
        putStrLn ("Iteration " ++ show i)
  iterateM mcnt nn ((dispAndInc >>) . online 0.000001 dataset)

dotest :: (Component n, Inp n ~ Sentence, Out n ~ Sentiment, Run n ~ IO)
       => n -> [(Inp n, Out n)] -> IO ()
dotest nn dataset = do
    putStrLn "Start test"
    result <- mapM ((>>= postprocess) . forward nn . fst) dataset
    expect <- mapM (postprocess . snd) dataset
    let (co,wr) = partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)
    putStrLn $ "First 10 tests:"
    flip mapM_ (take 10 dataset) $ \(ds,ev) -> do
      pv <- forward nn ds
      putStrLn $ showPretty $ text "+" <+> prettyDenseVectorFloat pv
      putStrLn $ showPretty $ text "*" <+> prettyDenseVectorFloat ev
  where
    postprocess :: Sentiment -> IO Int
    postprocess v = do
      a <- denseVectorToVector v
      return $ BV.maxIndex a

online :: (Component n, Inp n ~ Sentence, Out n ~ Sentiment, Run n ~ IO)
       => Float -> [(Inp n, Out n)] -> n -> IO n
online rate ds nn = walk ds nn
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

instance Pretty (DenseVector Float) where
  pretty = prettyDenseVectorFloat
showPretty x = displayS (renderPretty 0.4 500 x) ""
