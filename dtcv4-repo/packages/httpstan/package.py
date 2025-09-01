# Copyright Spack Project Developers. See COPYRIGHT file for details.
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
#     spack install httpstan
#
# You can edit this file again by typing:
#
#     spack edit httpstan
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class Httpstan(MakefilePackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://www.example.com"
    url = "https://github.com/stan-dev/httpstan/archive/refs/tags/4.13.0.tar.gz"

    # FIXME: Add a list of GitHub accounts to
    # notify when the package is updated.
    # maintainers("github_user1", "github_user2")

    # FIXME: Add the SPDX identifier of the project's license below.
    # See https://spdx.org/licenses/ for a list. Upon manually verifying
    # the license, set checked_by to your Github username.
    license("UNKNOWN", checked_by="github_user1")

    version("4.13.0", sha256="6b15a07557715e79e6fd66993930003b270f8b8b0c9e65f84978afe5e6bb3047")

    depends_on("cxx", type="build")

    depends_on('python', type=('build', 'run'))
    depends_on('py-setuptools', type=('build', 'run'))
    depends_on('py-poetry-core', type='build')
    depends_on('py-aiohttp', type=('build', 'run'))
    depends_on('py-appdirs', type=('build', 'run'))
    depends_on('py-webargs', type=('build', 'run'))
    depends_on('py-marshmallow', type=('build', 'run'))
    depends_on('py-numpy', type=('build', 'run'))
    depends_on('py-pybind11')
    depends_on('cmake', type=('build', 'run'))

    # FIXME: Add dependencies if required.
    # depends_on("foo")

    def edit(self, spec, prefix):
        # FIXME: Edit the Makefile if necessary
        # FIXME: If not needed delete this function
        makefile = FileFilter("Makefile")
        makefile.filter("CC = .*", "CC = cc")
        pass

    #def install(self, spec, prefix):
        #for thing in dir(self.stage):
        #    print(thing, getattr(self.stage,thing))
        #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(self.build_directory)
        #print(prefix)


        #glob_string = join_path(self.stage.path, "spack-build*/bin/Fall3d.x")

        #src = glob.glob(glob_string)[0]

        #src = join_path(
        #        self.stage.path,
        #              "spack-build-pio5mzu/bin/Fall3d.x"
        #        )
        #out_path = join_path(prefix,"bin")

        #os.mkdir(out_path)

        #destination = join_path(out_path,"Fall3d.x")

        #print("SRC:", src)
       # print("DESTINATION:", destination)

        #install(
        #        src, destination
        #        )
        #cmake("-DCMAKE_INSTALL_PREFIX:PATH=~/")
        #make()
       # make("install")

    #def cmake_args(self):
        # FIXME: Add arguments other than
        # FIXME: CMAKE_INSTALL_PREFIX and CMAKE_BUILD_TYPE
        # FIXME: If not needed delete this function
        #args = [elfi

        #return args

