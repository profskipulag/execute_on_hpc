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
#     spack install py-datapi
#
# You can edit this file again by typing:
#
#     spack edit py-datapi
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class PyDatapi(PythonPackage):
    """ESEE Data Stores API Python Client."""
    homepage = "https://github.com/ecmwf-projects/datapi"
    pypi     = "datapi/datapi-0.4.0.tar.gz"

    license("Apache-2.0")
    version("0.4.0", sha256="6355544b01a51192a87016368527fc7e55429f546ff786c6042a78876d3c497f")

    extends("python")
    depends_on("python@3.8:", type=("build", "run"))

    # PEP 517 backend + versioning
    depends_on("py-setuptools",      type="build")
    depends_on("py-setuptools-scm",   type="build")

    # Runtime deps from pyproject
    depends_on("py-attrs",                type=("build", "run"))
    depends_on("py-requests",             type=("build", "run"))
    depends_on("py-multiurl",      type=("build", "run"))
    depends_on("py-typing-extensions",    type=("build", "run"))

    # Optional extra: legacy CDS API client (controlled via a variant if you like)
    # variant("legacy", default=False, description="Enable legacy CDS API extra")
    # depends_on("py-cdsapi@0.7.5:", when="+legacy", type=("build", "run"))

    import_modules = ["datapi"]

