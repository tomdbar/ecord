import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

if sys.platform == 'linux':
    # On linux --> try to compile with OpenMP.
    def make_extension(name, source):
        # return Extension(name, [source], extra_compile_args = ['-fopenmp'], extra_link_args = ['-fopenmp']) # SLOW on JupyterHub
        return Extension(name, [source]) # Faster on JupyterHub
else: # On a mac, the platform in 'darwin'.
    # OpenMP not available, falling back.
    def make_extension(name, source):
        return Extension(name, [source])

ext_modules = [make_extension("ecord.environment._tradjectory_step_cy", "ecord/environment/_tradjectory_step_cy.pyx")]

setup(
    name="ecord",
    version="0.0.1",
    author="Thomas D Barrett",
    author_email="t.barrett@instadeep.com",
    description="Supporting code for ECORD combinatorial optimisation solver.",
    url="https://gitlab.com/tomdbar/ecord",
    packages=find_packages(include=["ecord"]),
    ext_modules = cythonize(ext_modules),
    include_dirs=[np.get_include()],
    setup_requires=['Cython'],
    zip_safe=False,
)