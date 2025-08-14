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
#     spack install py-pytensor
#
# You can edit this file again by typing:
#
#     spack edit py-pytensor
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class PyPytensor(PythonPackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://www.example.com"
    pypi = "pytensor/pytensor-2.30.3.tar.gz"

    # FIXME: Add a list of GitHub accounts to
    # notify when the package is updated.
    # maintainers("github_user1", "github_user2")

    # FIXME: Add the SPDX identifier of the project's license below.
    # See https://spdx.org/licenses/ for a list. Upon manually verifying
    # the license, set checked_by to your Github username.
    license("UNKNOWN", checked_by="github_user1")

    version("2.30.3", sha256="d9f591cea045c72df520d68677bf7422c425df3a48d3e3173914e04536cd7d7f")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    # FIXME: Only add the python/pip/wheel dependencies if you need specific versions
    # or need to change the dependency type. Generic python/pip/wheel dependencies are
    # added implicity by the PythonPackage base class.
    # depends_on("python@2.X:2.Y,3.Z:", type=("build", "run"))
    # depends_on("py-pip@X.Y:", type="build")
    # depends_on("py-wheel@X.Y:", type="build")

    # FIXME: Add a build backend, usually defined in pyproject.toml. If no such file
    # exists, use setuptools.
    depends_on("py-setuptools", type="build")
    depends_on("py-numpy", type="build")
    depends_on("py-versioneer", type="build")
    depends_on("py-scipy", type="build")
    depends_on("py-toml", type="build")
# from https://gitlab.ebrains.eu/rominabaila/ebrains-spack-builds/-/blob/cd0166f128c87972876117562009b1a59df19c69/packages/py-pytensor/package.py
   # depends_on("python@3.10:3.13", type=("build", "run"))
   # depends_on("py-setuptools@59.0.0:", type="build")
    depends_on("py-cython", type="build")
    #depends_on("py-versioneer+toml", type="build")
   # depends_on("py-scipy@1.0:1", type=("build", "run"))
   # depends_on("py-numpy@1.17.0:", type=("build", "run"))
    depends_on("py-filelock", type=("build", "run")) # TODO: it needs filelock>=3.15, but on pypi the latest one is 3.12.4
    depends_on("py-etuples", type=("build", "run"))
   # depends_on("py-logical-unification", type=("build", "run"))
   # depends_on("py-mini-kanren", type=("build", "run"))
    depends_on("py-cons", type=("build", "run"))

    # depends_on("py-hatchling", type="build")
    # depends_on("py-flit-core", type="build")
    # depends_on("py-poetry-core", type="build")

    # FIXME: Add additional dependencies if required.
    # depends_on("py-foo", type=("build", "run"))

    def config_settings(self, spec, prefix):
        # FIXME: Add configuration settings to be passed to the build backend
        # FIXME: If not needed, delete this function
        settings = {}
        return settings
