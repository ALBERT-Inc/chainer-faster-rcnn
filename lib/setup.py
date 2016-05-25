#https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/setup.py

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


# run the customize_compiler
#class custom_build_ext(build_ext):
#    def build_extensions(self):
#        customize_compiler_for_nvcc(self.compiler)
#        build_ext.build_extensions(self)



ext_modules = [
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        #extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    )
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    #cmdclass={'build_ext': custom_build_ext},
    cmdclass={'build_ext': build_ext},
)

