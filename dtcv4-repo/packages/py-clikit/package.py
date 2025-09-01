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
#     spack install py-clikit
#
# You can edit this file again by typing:
#
#     spack edit py-clikit
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class PyClikit(PythonPackage):
    """CliKit: utilities to build beautiful, testable CLIs."""
    homepage = "https://github.com/sdispater/clikit"
    pypi = "clikit/clikit-0.6.2.tar.gz"

    license("MIT")
    version("0.6.2", sha256="442ee5db9a14120635c5990bcdbfe7c03ada5898291f0c802f77be71569ded59")

    extends("python")

    # PEP 517 backend (required to import poetry.core.masonry.api)
    depends_on("py-poetry-core@1.0:", type="build")

    # Runtime deps (pinned by upstream metadata)
    depends_on("py-crashtest@0.3:0.3", type=("build", "run"))
    depends_on("py-pastel@0.2:0.2",     type=("build", "run"))
    depends_on("py-pylev@1.3:1",        type=("build", "run"))

    # Quick import check after install
    import_modules = ["clikit"]

