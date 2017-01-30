{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveDataTypeable #-}
module Main where

import qualified Data.Text as Text
import qualified Data.Text.Encoding as Text
import qualified Data.ByteString as BS
import qualified Data.Vector as BV
import Control.Monad.Except
import Data.Hashable
import Data.Data
import qualified NLP.Tokenize.Text as Token
import System.FilePath
import System.Directory
import Data.NeuralNetwork
import Data.NeuralNetwork.Adapter
import Data.NeuralNetwork.Backend.BLASHS

load_imdb_train = do
  let path = "tdata" </> "aclImdb" </> "train" </> "pos"
  files <- listDirectory path
  tr_pos <- mapM (load_entry path) files
  let path = "tdata" </> "aclImdb" </> "train" </> "neg"
  files <- listDirectory path
  tr_neg <- mapM (load_entry path) files
  return $ tr_pos ++ tr_neg

load_entry path file = do
  content <- BS.readFile (path </> file)
  let toks = Token.tokenize (Text.decodeUtf8 content)
      eval = let (_, '_':n) = break (=='_') file in read n :: Int
  return (map hash toks, eval)

main = do let cc = InString        :++ Flow (LSTM 10) :++
                   Cutoff 80       :++ Concat         :++
                   FullConnect 100 :++ FullConnect 10
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
            Left _    -> putStrLn "Error."
            Right cnn -> putStrLn "OK."

data SpecCutoff = Cutoff Int deriving (Typeable, Data)
type Cutoff = Adapter IO [DenseVector Float] [DenseVector Float] Int
cutoff :: Num a => Int -> Cutoff
cutoff n = Adapter to back
  where
    to inp = do
      let r = length inp
      pd <- replicateM (n-r) (newDenseVector (size $ head inp))
      return (r, take n $ inp ++ pd)
    back r odelta = do
      pd <- replicateM (r-n) (newDenseVector (size $ head odelta))
      return $ take r (odelta ++ pd)

instance BodySize SpecCutoff where
  bsize (SV s) (Cutoff n) = SF n s
  bsize (SF m s) (Cutoff n) = SF n s

instance TranslateBody SpecCutoff where
  --
  type SpecToCom SpecCutoff = Cutoff
  trans (SV (D1 _)) (Cutoff n) = return $ cutoff n
  trans _ _ = throwError ErrMismatch

data SpecConcat = Concat deriving (Typeable, Data)
type Concat = Adapter IO [DenseVector Float] (DenseVector Float) (Int, Int)
nconcat :: Concat
nconcat = Adapter to back
  where
    to inp = do let vinp = BV.fromList inp
                cv <- denseVectorConcat vinp
                return ((BV.length vinp, size (BV.head vinp)), cv)
    back (m,n) odelta = do let videlta = denseVectorSplit m n odelta
                           return $ BV.toList videlta

instance BodySize SpecConcat where
  bsize (SF m (D1 n)) Concat = D1 (m*n)

instance TranslateBody SpecConcat where
  type SpecToCom SpecConcat = Concat
  trans (SF m (D1 n)) Concat = return nconcat
  trans _ _ = throwError ErrMismatch
