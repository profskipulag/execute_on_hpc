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
#     spack install py-geopandas
#
# You can edit this file again by typing:
#
#     spack edit py-geopandas
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class PyGeopandas(PythonPackage):
    """GeoPandas: pandas + shapely for geospatial data."""

    homepage = "https://geopandas.org"
    pypi     = "geopandas/geopandas-1.1.1.tar.gz"

    license("BSD-3-Clause")
    version("1.1.1", sha256="1745713f64d095c43e72e08e753dbd271678254b24f2e01db8cdb8debe1d293d")

    extends("python")
    depends_on("python@3.10:", type=("build", "run"))

    # Build backend (fixes your error)
    depends_on("py-setuptools", type="build")

    # Required runtime deps for 1.1.x
    depends_on("py-packaging",          type=("build", "run"))
    depends_on("py-pandas@2.0:",        type=("build", "run"))
    depends_on("py-numpy@1.24:",        type=("build", "run"))
    depends_on("py-shapely@2.0:",       type=("build", "run"))
    depends_on("py-pyproj@3.5:",        type=("build", "run"))
    depends_on("py-pyogrio@0.7.2:",     type=("build", "run"))

    # Optional (common extras)
    # depends_on("py-matplotlib@3.7:",  type=("build", "run"), when="+plot")

    import_modules = ["geopandas"]

