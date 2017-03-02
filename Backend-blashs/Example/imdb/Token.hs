module Token where

import Data.Text
import NLP.Tokenize.Text

tokenize :: Text -> IO [Text]
tokenize = return . run whitespace
