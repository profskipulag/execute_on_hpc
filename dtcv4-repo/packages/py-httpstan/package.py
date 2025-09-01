from spack.package import *


class PyHttpstan(PythonPackage):
    """HTTP 1.1 interface to the Stan C++ library (server)."""

    homepage = "https://httpstan.readthedocs.org"
    url      = "https://github.com/stan-dev/httpstan/archive/refs/tags/4.13.0.tar.gz"

    version("4.13.0", sha256="6b15a07557715e79e6fd66993930003b270f8b8b0c9e65f84978afe5e6bb3047")

    extends("python")
    depends_on("python@3.10:", type=("build", "run"))

    # Build backend + helpers (Poetry Core + setuptools)
    depends_on("py-poetry-core@1.0:", type="build")
    depends_on("py-setuptools", type="build")  # needed by httpstan/build.py

    # Runtime/build requirements from upstream
    depends_on("py-aiohttp@3.8:", type=("build", "run"))
    depends_on("py-appdirs",      type=("build", "run"))
    depends_on("py-webargs",      type=("build", "run"))
    depends_on("py-marshmallow",  type=("build", "run"))
    depends_on("py-numpy",        type=("build", "run"))
    depends_on("py-pybind11",     type="build")

    # **NEW:** needed since we call make() below
    depends_on("gmake", type="build")

    # Toolchain constraints from upstream docs
    conflicts("%gcc@:8",   msg="httpstan requires gcc >= 9")
    conflicts("%clang@:9", msg="httpstan requires clang >= 10")

    # Upstream builds C/C++ libs with 'make' before building the wheel.
    # Run that step just before pip install.
    @run_before("install")
    def build_httpstan_libraries(self):
        make()  # uses the repository Makefile to produce required libraries

    # Sanity check after install
    import_modules = ["httpstan"]

