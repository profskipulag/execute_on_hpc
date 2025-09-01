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
#     spack install py-pysimdjson
#
# You can edit this file again by typing:
#
#     spack edit py-pysimdjson
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class PyPysimdjson(PythonPackage):
    """Python bindings for the simdjson project (SIMD-accelerated JSON parser)."""
    homepage = "https://pysimdjson.tkte.ch/"
    pypi = "pysimdjson/pysimdjson-7.0.2.tar.gz"

    # Dual-licensed upstream: MIT and Apache-2.0
    license("Apache-2.0", "MIT")

    version("7.0.2", sha256="44cf276e48912a3b9c7ca362c14da8420a7ac15a9f1a16ec95becff86db3904a")

    # Build toolchain
    depends_on("cxx", type="build")
    depends_on("py-setuptools", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-cython", type="build")  # <-- required to generate csimdjson.* from .pyx

    # Optional: tell Spack to import the installed module for sanity check
    import_modules = ["simdjson"]

