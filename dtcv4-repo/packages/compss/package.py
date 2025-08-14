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
#     spack install compss
#
# You can edit this file again by typing:
#
#     spack edit compss
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class Compss(Package):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://www.example.com"
    url = "https://compss.bsc.es/repo/sc/stable/COMPSs_3.3.2.tar.gz"

    # FIXME: Add a list of GitHub accounts to
    # notify when the package is updated.
    # maintainers("github_user1", "github_user2")

    # FIXME: Add the SPDX identifier of the project's license below.
    # See https://spdx.org/licenses/ for a list.
    license("UNKNOWN")

    version("3.3.2", sha256="7dffd1e5c7cedf23b65b523fc4daefff3be8c04c788a0e116039d07585769a0e")

    # FIXME: Add dependencies if required.
    # depends_on("foo")

    # FIXME: Add dependencies if required.
    depends_on('python')
    depends_on('py-setuptools', type='build')
    depends_on('openjdk', type='build')
    depends_on('boost')
    depends_on('libxml2')
    depends_on('gradle', type='build')
    depends_on('autoconf', type='build')
    depends_on('automake', type='build')
    depends_on('libtool', type='build')
    depends_on('m4', type='build')
    depends_on('py-pip')
    depends_on('py-wheel')

    def install(self, spec, prefix):
        # FIXME: Unknown build system
        import os
        print("Prefix: " + str(prefix))
        install_script = Executable('./install')
        install_script('-T', prefix.compss)
        print("Dirs: " +str(os.listdir(str(prefix))))


    def setup_run_environment(self, env):
        env.set('COMPSS_HOME', self.prefix.compss)
        env.prepend_path('PATH', self.prefix.compss + '/Runtime/scripts/user')

