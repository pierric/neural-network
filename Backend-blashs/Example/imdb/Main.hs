{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE FlexibleInstances #-}
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
import Text.PrettyPrint.Free hiding ((</>))
import Data.NeuralNetwork hiding (cost')
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

load_imdb_test = do
  let path = "tdata" </> "aclImdb" </> "test" </> "pos"
  files <- listDirectory path
  tr_pos <- mapM (load_entry path) files
  let path = "tdata" </> "aclImdb" </> "test" </> "neg"
  files <- listDirectory path
  tr_neg <- mapM (load_entry path) files
  return $ tr_pos ++ tr_neg

load_entry :: FilePath -> FilePath -> IO ([DenseVector Float], DenseVector Float)
load_entry path file = do
  content <- BS.readFile (path </> file)
  let toks = Token.tokenize (Text.decodeUtf8 content)
      eval = let (_, '_':m) = break (=='_') file
                 (n, _) = break (=='.') m
             in read n :: Int
  toks' <- mapM (newDenseVectorConst 1 . (/16777216.0) . fromIntegral . hash) toks
  eval' <- DenseVector <$> (SV.unsafeThaw $ SV.fromList (replicate eval 0 ++ [1] ++ replicate (9-eval) 0))
  return (toks', eval')

main = do let cc = InStream :++ -- Debug "i0" :++
                   Flow (LSTM 2) :++ -- Debug "i1" :++
                   Cutoff 80 :++ Concat :++
                   FullConnect 10
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
              putStrLn "1st line."
              let tr0@(i,e) = head trdata
              let rate  = 0.1
              o <- forward nn i
              putStrLn $ showPretty $ text "#" <+> prettyDenseVectorFloat o
              nn <- learn diff rate nn tr0
              o <- forward nn i
              putStrLn $ showPretty $ text "#" <+> prettyDenseVectorFloat o
              nn <- learn diff rate nn tr0
              o <- forward nn i
              putStrLn $ showPretty $ text "#" <+> prettyDenseVectorFloat o
              nn <- learn diff rate nn tr0
              o <- forward nn i
              putStrLn $ showPretty $ text "#" <+> prettyDenseVectorFloat o
              nn <- learn diff rate nn tr0
              o <- forward nn i
              putStrLn $ showPretty $ text "#" <+> prettyDenseVectorFloat o
              putStrLn $ showPretty $ text "%" <+> prettyDenseVectorFloat e
  where
     diff a b = do v <- newDenseVector (size a)
                   v <<= ZipWith cost' a b
                   return v

data SpecDebug a = Debug String deriving (Typeable, Data)
type Debug a = Adapter IO a a ()
instance BodySize (SpecDebug a) where
  bsize s (Debug _) = s
instance Pretty a => TranslateBody (SpecDebug a) where
  type SpecToCom (SpecDebug a) = Debug a
  trans s (Debug name)= return $ Adapter to back
    where
      to inp = do putStrLn $ (name ++ "-(Forward):" )
                  putStrLn $ showPretty $ indent 2 $ pretty inp
                  return ((), inp)
      back _ odelta = return odelta

data SpecCutoff = Cutoff Int deriving (Typeable, Data)
type Cutoff = Adapter IO [DenseVector Float] [DenseVector Float] Int

instance BodySize SpecCutoff where
  bsize (SV (D1 s)) (Cutoff n) = SF n (D1 s)

instance TranslateBody SpecCutoff where
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

instance TranslateBody SpecConcat where
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
