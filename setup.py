from glob import glob
from os.path import dirname, join as pjoin
import sys
from setuptools import setup, find_packages
from setup_helpers import read_vars_from

PACKAGES = find_packages()

# Get version and release info, which is all stored in cupyimg/version.py
info = read_vars_from(pjoin('cudipy', 'info.py'))

# We may just have imported setuptools, or we may have been exec'd from a
# setuptools environment like pip
using_setuptools = 'setuptools' in sys.modules
extra_setuptools_args = {}
if using_setuptools:
    # Try to preempt setuptools monkeypatching of Extension handling when Pyrex
    # is missing.  Otherwise the monkeypatched Extension will change .pyx
    # filenames to .c filenames, and we probably don't have the .c files.
    sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
    # Set setuptools extra arguments
    extra_setuptools_args = dict(
        tests_require=['pytest'],
        zip_safe=False,
        extras_require=info.EXTRAS_REQUIRE,
        python_requires=">= 3.6",
    )

# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 24.2.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []


def main(**extra_args):
    setup(
        name=info.NAME,
        maintainer=info.MAINTAINER,
        maintainer_email=info.MAINTAINER_EMAIL,
        description=info.DESCRIPTION,
        long_description=info.LONG_DESCRIPTION,
        url=info.URL,
        download_url=info.DOWNLOAD_URL,
        license=info.LICENSE,
        classifiers=info.CLASSIFIERS,
        author=info.AUTHOR,
        author_email=info.AUTHOR_EMAIL,
        platforms=info.PLATFORMS,
        version=info.VERSION,
        install_requires=info.REQUIRES,
        requires=info.REQUIRES,
        provides=info.PROVIDES,
        packages=PACKAGES,
        setup_requires=SETUP_REQUIRES,
        data_files=[('share/doc/cudipy/examples',
                     glob(pjoin('doc', 'examples', '*.py')))],
        **extra_args,
    )


if __name__ == "__main__":
    main(**extra_setuptools_args)
