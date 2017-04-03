import Distribution.Simple
import Distribution.Simple.Setup
import Distribution.Simple.LocalBuildInfo
import Distribution.Simple.Program
import Distribution.Simple.Program.Run
import Distribution.Simple.Program.Ar
import Distribution.Simple.BuildPaths
import Distribution.System
import Distribution.Verbosity
import Distribution.PackageDescription
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
  dir <- filterM (doesDirectoryExist . (</> "cbits")) dirs
  if (null dir)
    then
      buildHook simpleUserHooks pkgdesc binfo uh bf
    else do
      let sdir = head dir
          ssrc = sdir </> "cbits" </> (vecmod ++ ".ll")
          verb = fromFlagOrDefault normal (buildVerbosity bf)
      runProgramInvocation verb $ simpleProgramInvocation "llc" ["-filetype=obj", "-o=" ++ sobj, ssrc]
      -- generate a ".a" archive for the executable
      let slib = buildroot </> ("lib" ++ vecmod ++ ".a")
      createArLibArchive verb binfo slib [sobj]
      extralib <- canonicalizePath buildroot
      let pkgdesc' = updatePackageDescription (Nothing, [("t1", libBI)]) pkgdesc
          libBI    = emptyBuildInfo {extraLibs = [vecmod], extraLibDirs = [extralib]}
      buildHook simpleUserHooks pkgdesc' binfo uh bf
      -- however the library is static and doesn't include the vecmod
      -- we will then explicitly to insert it.
      let unitId  = componentUnitId (getComponentLocalBuildInfo binfo CLibName)
          vlibPath = buildroot </> mkLibName     unitId
          plibPath = buildroot </> mkProfLibName unitId
          whenVanillaLib = when (withVanillaLib binfo)
          whenProfLib    = when (withProfLib binfo)
          Platform hostArch hostOS = hostPlatform binfo
          args    = case hostOS of
                       OSX -> ["-q", "-s"]
                       _   -> ["-q"]
      (ar, _) <- requireProgram verb arProgram (withPrograms binfo)
      whenVanillaLib $
        runProgramInvocation verb $ programInvocation ar (args ++ [vlibPath, sobj])
      whenProfLib $
        runProgramInvocation verb $ programInvocation ar (args ++ [plibPath, sobj])
