module Hmatrix (
  v2hv, hv2v, v2hm, hm2v, t2hv,
  (<#), (#>), ident, dispf,
) where

import Data.Tensor
import qualified Numeric.LinearAlgebra as HM
import Numeric.LinearAlgebra ((<#), (#>), ident, dispf)
import qualified Data.Vector.Storable as PV
import System.IO.Unsafe (unsafePerformIO)


v2hv :: D1 -> PV.Vector Float -> HM.Vector Float
v2hv (D1 n) v = v

hv2v :: HM.Vector Float -> PV.Vector Float
hv2v v = v

v2hm :: D2 -> PV.Vector Float -> HM.Matrix Float
v2hm (D2 r c) v = HM.reshape c v

hm2v :: HM.Matrix Float -> PV.Vector Float
hm2v m = HM.flatten m

t2hv :: Tensor d Float -> HM.Vector Float
t2hv t = unsafePerformIO $ PV.unsafeFreeze (_tdat t)
