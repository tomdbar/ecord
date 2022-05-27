from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

import sys

# Build with : python setup_cy.py build_ext --inplace --force

if sys.platform == 'linux':
    # On linux --> try to compile with OpenMP.
    def make_extension(name, source):
        # return Extension(name, [source], extra_compile_args = ['-fopenmp'], extra_link_args = ['-fopenmp']) # SLOW on JupyterHub
        return Extension(name, [source]) # Faster on JupyterHub
else: # On my mac, the platform in 'darwin'.
    # OpenMP not available, falling back.
    def make_extension(name, source):
        return Extension(name, [source])

ext_modules = [make_extension("ecord.environment._tradjectory_step_cy", "ecord/environment/_tradjectory_step_cy.pyx")]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[np.get_include()],
    setup_requires=['Cython'],
    zip_safe=False
)