from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
package = Extension('pairwise_fast', ['pairwise_fast.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))