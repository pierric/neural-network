module Token where

import qualified Data.Text as Text
import Control.Monad
import NLP.Tokenize.Text

tokenize :: Text.Text -> IO [Text.Text]
tokenize text = do
  return $ run whitespace text
