{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
module Main where

import qualified Data.Vector as BV
import Control.Monad
import Control.Monad.Except
import Control.Monad.IO.Class
import Data.Data
import System.IO (hFlush, stdout)
import Data.IORef
import Data.List (partition)
import Text.Printf (printf)
import Text.PrettyPrint hiding ((</>))
import Text.PrettyPrint.HughesPJClass
import Blas.Generic.Unsafe (Numeric)
import Data.NeuralNetwork
import Data.NeuralNetwork.Adapter
import Data.NeuralNetwork.Backend.BLASHS
import Corpus

main = do putStrLn "Start."
          (nv, trdata, tsdata) <- corpus 1000
          let cc = Embedding nv :&: Flow (LSTM 128) :&: Cutoff 80 :&: Concat :&: FullConnect 10 :&: HNil
          x <- runExceptT $ compile byBLASHSd (InStream 1, cc, MeanSquaredError)
          case x of
            Left _   -> putStrLn "Error."
            Right nn -> do
              putStrLn "Loaded."
              loop nn (BV.take 200 trdata) (BV.take 20 trdata) 1
  where
     loop nn trd tsd cnt = do
       nn <- dotrain nn trd cnt
       dotest nn tsd
       putStr "Continue? (number):"
       hFlush stdout
       str <- getLine
       let next = (reads :: ReadS Int) str
       when (not $ null next) (loop nn trd tsd (fst $ head next))

dotrain :: (ModelCst n e, Run n ~ IO, Inp n ~ Sentence, Val e ~ Sentiment)
        => (n, e) -> BV.Vector (Inp n, Out n) -> Int -> IO (n,e)
dotrain nn dataset mcnt = do
  cnt <- newIORef 0 :: IO (IORef Int)
  let dispAndInc = do
        i <- readIORef cnt
        writeIORef cnt (i+1)
        putStrLn ("Iteration " ++ show i)
  iterateM mcnt nn ((dispAndInc >>) . online 0.0001 dataset)

dotest :: (ModelCst n e, Run n ~ IO, Inp n ~ Sentence, Val e ~ Sentiment)
       => (n, e) -> BV.Vector (Inp n, Out n) -> IO ()
dotest (nn,_) dataset = do
    putStrLn "Start test"
    result <- BV.mapM ((>>= postprocess) . forward nn . fst) dataset
    expect <- BV.mapM (postprocess . snd) dataset
    let (co,wr) = BV.partition (uncurry (==)) $ BV.zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (BV.length co) (BV.length wr)
    putStrLn $ "First 10 tests:"
    BV.forM_ (BV.take 10 dataset) $ \(ds,ev) -> do
      pv <- forward nn ds
      putStrLn $ showPretty $ text "+" <+> pPrint pv
      putStrLn $ showPretty $ text "*" <+> pPrint ev
  where
    postprocess :: Sentiment -> IO Int
    postprocess v = do
      a <- denseVectorToVector v
      return $ BV.maxIndex a

online :: (ModelCst n e, Run n ~ IO, Inp n ~ Sentence, Val e ~ Sentiment)
       => Float -> BV.Vector (Inp n, Out n) -> (n, e) -> IO (n, e)
online rate ds nn = walk (BV.toList ds) nn
  where
    walk []     nn = return nn
    walk (d:ds) nn = do nn <- learn nn d rate
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

data SpecCutoff = Cutoff Int deriving (Typeable, Data)
type Cutoff p = Adapter IO [DenseVector p] [DenseVector p] Int

instance BodySize SpecCutoff where
  bsize (SV (D1 s)) (Cutoff n) = SF n (D1 s)

instance (MonadError ErrCode m, Numeric p, RealType p, SIMDable p) => BodyTrans m (ByBLASHS p) SpecCutoff where
  --
  type SpecToCom (ByBLASHS p) SpecCutoff = Cutoff p
  btrans _ (SV (D1 s)) (Cutoff n) = return $ cutoff n
    where
      cutoff n = Adapter to back
      to inp = do
        let r = length inp
        z <- replicateM (n-r) (newDenseVector s)
        return (r, take n $ inp ++ z)
      back r odelta = do
        z <- replicateM (r-n) (newDenseVector s)
        return $ take r (odelta ++ z)
  btrans _ _ _ = throwError ErrMismatch

data SpecConcat = Concat deriving (Typeable, Data)
type Concat p = Adapter IO [DenseVector p] (DenseVector p) (Int, Int)

instance BodySize SpecConcat where
  bsize (SF m (D1 n)) Concat = D1 (m*n)

instance (MonadError ErrCode m, Numeric p, RealType p, SIMDable p) => BodyTrans m (ByBLASHS p) SpecConcat where
  type SpecToCom (ByBLASHS p) SpecConcat = Concat p
  btrans _ (SF m (D1 n)) Concat = return nconcat
    where
      nconcat = Adapter to back
      to inp = do let vinp = BV.fromList inp
                  cv <- denseVectorConcat vinp
                  return ((BV.length vinp, size (BV.head vinp)), cv)
      back (m,n) odelta = do let videlta = denseVectorSplit m n odelta
                             return $ BV.toList videlta
  btrans _ _ _ = throwError ErrMismatch

data SpecEmbedding = Embedding Int deriving (Typeable, Data)
type Embedding p = Adapter IO [Int] [DenseVector p] ()

instance BodySize SpecEmbedding where
  bsize (SV (D1 1)) (Embedding n)= SV (D1 n)

instance (MonadError ErrCode m, Numeric p, RealType p, SIMDable p) => BodyTrans m (ByBLASHS p) SpecEmbedding where
  type SpecToCom (ByBLASHS p) SpecEmbedding = Embedding p
  btrans _ (SV (D1 s)) (Embedding n) = return $ Adapter to back
    where
      to inp = do v <- mapM (newDenseVectorDelta n) inp
                  return ((), v)
      back () odelta = return $ repeat 0
      newDenseVectorDelta n i = do
        v <- newDenseVector n
        unsafeWriteV v i 1
        return v

showPretty = renderStyle (Style PageMode 500 0.4)
