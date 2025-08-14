# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install fall3d
#
# You can edit this file again by typing:
#
#     spack edit fall3d
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *
import os
import glob

class Fall3d(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://www.example.com"
    url = "https://gitlab.com/fall3d-suite/fall3d/-/archive/master/fall3d-master.tar.gz"

    # FIXME: Add a list of GitHub accounts to
    # notify when the package is updated.
    # maintainers("github_user1", "github_user2")

    # FIXME: Add the SPDX identifier of the project's license below.
    # See https://spdx.org/licenses/ for a list.
    license("UNKNOWN")

    version("3d-master", sha256="71e6b0b6ddefae5bafd1b717f0e9acf043f30cb86af8c1050be3073a59202445")

    # FIXME: Add dependencies if required.
    depends_on("netcdf-fortran")

    def install(self, spec, prefix):
        for thing in dir(self.stage):
            print(thing, getattr(self.stage,thing))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(self.build_directory)
        print(prefix)


        glob_string = join_path(self.stage.path, "spack-build*/bin/Fall3d.x")

        src = glob.glob(glob_string)[0]

        #src = join_path(
        #        self.stage.path,
        #              "spack-build-pio5mzu/bin/Fall3d.x"
        #        )
        out_path = join_path(prefix,"bin")

        os.mkdir(out_path)

        destination = join_path(out_path,"Fall3d.x")

        print("SRC:", src)
        print("DESTINATION:", destination)

        install(
                src, destination
                )
        #cmake("-DCMAKE_INSTALL_PREFIX:PATH=~/")
        #make()
        #make("install")

    #def cmake_args(self):
        # FIXME: Add arguments other than
        # FIXME: CMAKE_INSTALL_PREFIX and CMAKE_BUILD_TYPE
        # FIXME: If not needed delete this function
        #args = [elfi
        
        #return args
