module Parser where
import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Internal as BI
import Codec.Compression.GZip (decompress)
import Control.Monad
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV

type Pixel = Word8
type Image = SV.Vector Float
type Label = SV.Vector Float

decodeImages :: Get [Image]
decodeImages = do
    mc <- getWord32be
    guard (mc == 0x00000803)
    [d1,d2,d3] <- many 3 getWord32be
    guard (d2 == 28 && d3 == 28)
    many d1 pic
  where
    pic :: Get Image
    pic = do
      bs <- getByteString (28*28)
      return . toVecDouble . fromByteString $ bs
    toVecDouble :: SV.Vector Pixel -> SV.Vector Float
    toVecDouble = SV.map ((/255) . fromIntegral)
    fromByteString :: BS.ByteString -> SV.Vector Pixel
    fromByteString (BI.PS ptr o n) = SV.unsafeFromForeignPtr ptr o n

decodeLabels :: Get [Label]
decodeLabels = do
    mc <- getWord32be
    guard (mc == 0x00000801)
    d1 <- getWord32be
    many d1 lbl
  where
    lbl :: Get Label
    lbl = do
      v <- fromIntegral <$> (get :: Get Word8)
      return $ SV.fromList (replicate v 0 ++ [1] ++ replicate (9-v) 0)

many :: (Integral n, Monad m) => n -> m a -> m [a]
many cnt dec = sequence (replicate (fromIntegral cnt) dec)

trainingData :: IO ([Image], [Label])
trainingData = do
    s <- decompress <$> BL.readFile "data/train-images-idx3-ubyte.gz"
    t <- decompress <$> BL.readFile "data/train-labels-idx1-ubyte.gz"
    return (runGet decodeImages s, runGet decodeLabels t)

testData :: IO ([Image], [Label])
testData = do
    s <- decompress <$> BL.readFile "data/t10k-images-idx3-ubyte.gz"
    t <- decompress <$> BL.readFile "data/t10k-labels-idx1-ubyte.gz"
    return (runGet decodeImages s, runGet decodeLabels t)
