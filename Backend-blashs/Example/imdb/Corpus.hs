{-# LANGUAGE BangPatterns #-}
module Corpus (corpus, Sentence, Sentiment) where

import Data.IORef
import qualified Data.Text as Text
import qualified Data.Text.Encoding as Text
import qualified Data.ByteString as BS
import qualified Data.Vector as BV
import qualified Data.Vector.Mutable as MV
import qualified Data.Vector.Storable as SV
import qualified Data.Map.Strict as Map
import Data.List (sortOn)
import Control.Monad.State.Strict
import System.FilePath
import System.Directory
import Control.Monad
import Data.Maybe (isNothing, fromMaybe)
import System.Random.MWC
import Data.NeuralNetwork.Backend.BLASHS
import Token

type Sentence  = [Int]
type Sentiment = DenseVector Float

slice = id -- (take 6000 <$>)

corpus :: Int -> IO (Int, BV.Vector (Sentence, Sentiment), BV.Vector (Sentence, Sentiment))
corpus wordmax = do
  train <- load_imdb_train
  test  <- load_imdb_test
  let dict@(mapp,num) = build_dict (word_freq $ train BV.++ test) wordmax 0
  print mapp
  putStrLn $ "Num of words:" ++ show num
  train <- BV.mapM (assign dict) train >>= shuffle
  test  <- BV.mapM (assign dict) test  >>= shuffle
  return (num, train, test)

load_imdb_train :: IO (BV.Vector ([Text.Text], Int))
load_imdb_train = do
  let path = "tdata" </> "aclImdb" </> "train" </> "pos"
  files <- slice $ listDirectory path
  tr_pos <- BV.mapM (load_entry path) $ BV.fromList files
  let path = "tdata" </> "aclImdb" </> "train" </> "neg"
  files <- slice $ listDirectory path
  tr_neg <- BV.mapM (load_entry path) $ BV.fromList files
  return $ tr_pos BV.++ tr_neg

load_imdb_test :: IO (BV.Vector ([Text.Text], Int))
load_imdb_test = do
  let path = "tdata" </> "aclImdb" </> "test" </> "pos"
  files <- slice $ listDirectory path
  tr_pos <- BV.mapM (load_entry path) $ BV.fromList files
  let path = "tdata" </> "aclImdb" </> "test" </> "neg"
  files <- slice $ listDirectory path
  tr_neg <- BV.mapM (load_entry path) $ BV.fromList files
  return $ tr_pos BV.++ tr_neg

load_entry :: FilePath -> FilePath -> IO ([Text.Text], Int)
load_entry path file = do
  content <- BS.readFile (path </> file)
  toks    <- tokenize $ Text.decodeUtf8 content
  let eval = let (_, '_':m) = break (=='_') file
                 (n, _) = break (=='.') m
             in read n - 1 :: Int
  return (map Text.toLower toks, eval)

word_freq :: BV.Vector ([Text.Text], Int) -> Map.Map Text.Text Int
word_freq dat = flip execState Map.empty $
                  BV.forM_ dat (\(ws, _) ->
                    forM_ ws   (\w ->
                      modify' (Map.insertWith (+) w 1)))

build_dict :: Map.Map Text.Text Int -> Int -> Int -> (Map.Map Text.Text Int, Int)
build_dict freq nb_words nb_skip =
  let dict = take nb_words $ drop nb_skip $ reverse $ sortOn snd $ Map.toList freq
      dictWithIndex = zip (map fst dict) [1..]
      !cnt = snd $ last dictWithIndex
  in (Map.fromList dictWithIndex, cnt)

assign :: (Map.Map Text.Text Int, Int) -> ([Text.Text], Int) -> IO (Sentence, Sentiment)
assign (dict, num) (!ws,!e) = do
  let ws' = map (\w -> fromMaybe 0 $ Map.lookup w dict) ws
  eval <- newDenseVector 10
  unsafeWriteV eval e 1
  return (ws', eval)

shuffle :: BV.Vector a -> IO (BV.Vector a)
shuffle v = do
  mv  <- BV.unsafeThaw v
  gen <- createSystemRandom
  go gen mv (MV.length mv)
  BV.unsafeFreeze mv
  where
    go g v 0 = return ()
    go g v i = do
      r <- uniformR (0, i') g
      MV.swap v i' r
      go g v i'
      where
        i' = i - 1
