from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    name='vis',
    ext_modules=cythonize("visualizations/load_point_cloud.pyx"),
    include_dirs=[numpy.get_include()]
)
