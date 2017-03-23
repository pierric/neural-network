------------------------------------------------------------
-- |
-- Module      :  Data.NeuralNetwork.Debug
-- Description :  Neural network in abstract
-- Copyright   :  (c) 2016 Jiasen Wu
-- License     :  BSD-style (see the file LICENSE)
-- Maintainer  :  Jiasen Wu <jiasenwu@hotmail.com>
-- Stability   :  experimental
-- Portability :  portable
--
--
-- This module defines a layer that does nothing but
-- print of the input.
------------------------------------------------------------
{-# LANGUAGE UndecidableInstances #-}
module Data.NeuralNetwork.Debug where

import Data.Data
import Control.Monad.Trans (liftIO, MonadIO)
import Control.Monad.Except (MonadError)
import Text.PrettyPrint.Free hiding ((</>))
import Data.NeuralNetwork.Common
import Data.NeuralNetwork.Adapter

data SpecDebug a = Debug String deriving (Typeable, Data)

type Debug a = Adapter IO a a ()

instance BodySize (SpecDebug a) where
  bsize s (Debug _) = s

instance (Pretty a, MonadIO (Env b), MonadError ErrCode (Env b)) => BodyTrans b (SpecDebug a) where
  type SpecToCom b (SpecDebug a) o = Debug a
  btrans b s (Debug name) o = return $ Adapter to back
    where
      to inp = do liftIO $ putStrLn $ (name ++ "-(Forward):" )
                  liftIO $ putStrLn $ showPretty $ indent 2 $ pretty inp
                  return ((), inp)
      back _ odelta = return odelta

showPretty x = displayS (renderPretty 0.4 500 x) ""
