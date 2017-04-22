module Hmatrix (
  v2hv, hv2v, v2hm, hm2v, t2hv,
  (<#), (#>), (<>), outer, ident, tr', scale, dispf,
) where

import Data.Tensor
import qualified Numeric.LinearAlgebra as HM
import Numeric.LinearAlgebra ((<#), (#>), (<>), outer, ident, tr', scale, dispf)
import qualified Data.Vector.Storable as PV
import System.IO.Unsafe (unsafePerformIO)


v2hv :: (HM.Element a, Element a) => D1 -> PV.Vector a -> HM.Vector a
v2hv (D1 n) v = v

hv2v :: (HM.Element a, Element a) => HM.Vector a -> PV.Vector a
hv2v v = v

v2hm :: (HM.Element a, Element a) => D2 -> PV.Vector a -> HM.Matrix a
v2hm (D2 r c) v = HM.reshape c v

hm2v :: (HM.Element a, Element a) => HM.Matrix a -> PV.Vector a
hm2v m = HM.flatten m

t2hv :: (HM.Element a, Element a) => Tensor d a -> HM.Vector a
t2hv t = unsafePerformIO $ PV.unsafeFreeze (_tdat t)
