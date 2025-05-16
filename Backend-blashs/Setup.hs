import Distribution.Simple
import Distribution.Simple.Setup
import Distribution.Simple.LocalBuildInfo
import Distribution.Simple.Program
import Distribution.Simple.Program.Run
import Distribution.Simple.Program.Ar
import Distribution.Simple.BuildPaths
import Distribution.System
import Distribution.Utils.Path
import Distribution.Verbosity
import Distribution.PackageDescription
import Data.String (fromString)
import System.FilePath ( (</>) )
import System.Directory( canonicalizePath, doesDirectoryExist )
import Control.Monad   ( when, filterM )

main = defaultMainWithHooks simpleUserHooks { buildHook  = myBuildHook }

myBuildHook :: PackageDescription -> LocalBuildInfo -> UserHooks -> BuildFlags -> IO ()
myBuildHook pkgdesc binfo uh bf = do
  let buildroot = buildDir binfo
      vecmod = "compare"
      sobj = buildroot </> (vecmod ++ ".o")
      Just dirs = hsSourceDirs . libBuildInfo <$> library pkgdesc
  dir <- filterM (doesDirectoryExist . (</> "cbits") . getSymbolicPath) dirs
  case dir of
    [] -> buildHook simpleUserHooks pkgdesc binfo uh bf
    cbitsDir:xs -> do
      when (not (null xs)) (print "WARNING: found extra cbits folders?")

      let sdir = getSymbolicPath $ cbitsDir
          ssrc = sdir </> "cbits" </> (vecmod ++ ".ll")
          verb = fromFlagOrDefault normal (buildVerbosity bf)
      runProgramInvocation verb $ simpleProgramInvocation "llc" ["-filetype=obj", "-o=" ++ sobj, ssrc]
      -- generate a ".a" archive for the executable
      let slib = buildroot </> ("lib" ++ vecmod ++ ".a")
      createArLibArchive verb binfo slib [sobj]
      extralib <- canonicalizePath buildroot

      -- append the extra obj file for the dynamic library
      let extra_bi = emptyBuildInfo {extraLibs = [vecmod], extraLibDirs = [extralib]}
          pkgdesc' = updatePackageDescription (Just extra_bi, []) pkgdesc

      buildHook simpleUserHooks pkgdesc' binfo uh bf

      -- however the library is static and doesn't include the vecmod
      -- we will then explicitly to insert it.
      withAllComponentsInBuildOrder pkgdesc' binfo $ \comp compbi -> do
        case comp of 
          CLib _ -> do
            let unitId   = componentUnitId compbi
                vlibPath = buildroot </> mkLibName     unitId
                plibPath = buildroot </> mkProfLibName unitId

            let whenVanillaLib = when (withVanillaLib binfo)
            let whenProfLib    = when (withProfLib binfo)
            let Platform hostArch hostOS = hostPlatform binfo
            let args = case hostOS of
                         OSX -> ["-q", "-s"]
                         _   -> ["-q"]
            (ar, _) <- requireProgram verb arProgram (withPrograms binfo)
            whenVanillaLib $
              runProgramInvocation verb $ programInvocation ar (args ++ [vlibPath, sobj])
            whenProfLib $
              runProgramInvocation verb $ programInvocation ar (args ++ [plibPath, sobj])

          _ -> return ()
