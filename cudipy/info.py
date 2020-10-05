""" This file contains defines parameters for cuDIPY that we use to fill
settings in setup.py, the DIPY top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import cudipy
"""

# cuDIPY version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = 'dev'
# _version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = 'GPU accelerated diffusion MRI utilities in python'

# Note: this long_description is actually a copy/paste from the top-level
# README.rst, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
======
cuDIPY
======

cuDIPY is a python toolbox containing GPU-based implementations of algorithms
from the DIPY library.

cuDIPY is for research only; please do not use results from cuDIPY for
clinical decisions.

Website
=======

Information on the upstream DIPY project is available on the DIPY website -
https://dipy.org

Mailing Lists
=============

Please see the developer's list at
http://mail.scipy.org/mailman/listinfo/nipy-devel

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for the main DIPY package.
* Download as a tar/zip file the `current trunk`_.

.. _main repository: http://github.com/dipy/cudipy
.. _Documentation: http://dipy.org
.. _current trunk: https://github.com/dipy/cudipy/archive/master.zip

License
=======

cuDIPY is licensed under the terms of the BSD license.
Please see the LICENSE file in the dipy distribution.

cuDIPY uses other libraries also licensed under the BSD or the MIT licenses.
"""

# versions for dependencies
# Check these versions against .travis.yml and requirements.txt
CUPY_MIN_VERSION = '7.8.0'
NIBABEL_MIN_VERSION = '3.0.0'
NUMPY_MIN_VERSION = '1.12.0'
SCIPY_MIN_VERSION = '1.0'
DIPY_MIN_VERSION = '1.2.0'

# Main setup parameters
NAME                = 'cudipy'
MAINTAINER          = "Gregory R. Lee"
MAINTAINER_EMAIL    = "neuroimaging@python.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://dipy.org"
DOWNLOAD_URL        = "http://github.com/dipy/cudipy/archives/master"
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "cuDIPY developers"
AUTHOR_EMAIL        = "neuroimaging@python.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
PROVIDES            = ["cudipy"]
REQUIRES            = ["cupy (>=%s)" % CUPY_MIN_VERSION,
                       "dipy (>=%s)" % DIPY_MIN_VERSION,
                       "nibabel (>=%s)" % NIBABEL_MIN_VERSION,
                       "numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION]
EXTRAS_REQUIRE = {
    "test": [
        "pytest",
        "coverage",
        "coveralls",
        "codecov",
    ],
    "doc": [
        "cupy",
        "numpy",
        "scipy",
        "dipy",
        "nibabel>=3.0.0",
        "matplotlib",
    ],
    "viz": [
        "matplotlib"
    ],
}

EXTRAS_REQUIRE["all"] = list(set([a[i] for a in list(EXTRAS_REQUIRE.values())
                                  for i in range(len(a))]))
