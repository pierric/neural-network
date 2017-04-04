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
      -- modify each of the test suits, so that the LL code is linked in.
      let pkgdesc' = updatePD (vecmod, extralib) pkgdesc
      buildHook simpleUserHooks pkgdesc' binfo uh bf

updatePD :: (String, String) -> PackageDescription -> PackageDescription
updatePD (extraLib, extraDir) p
    = p{ library     = updateLibrary     (library     p)
      --  , executables = updateExecutables (executables p)
       }
    where
      bi = emptyBuildInfo {extraLibs = [extraLib], extraLibDirs = [extraDir]}
      updateLibrary     = fmap (\lib -> lib {libBuildInfo = bi `mappend` libBuildInfo lib})
      -- updateExecutables = map  (\exe -> exe {buildInfo = bi `mappend` buildInfo exe})
