import sys
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'pytinybvh',
        sources=['src/pytinybvh.cpp'],
        include_dirs=[
            pybind11.get_include(),
            'deps/',
        ],
        language='c++',
        extra_compile_args=(
            ['-std=c++20', '-O3', '-Wno-unused-variable']
            if sys.platform != 'win32'
            else ['/std:c++20', '/O2']
        )
    )
]

setup(ext_modules=ext_modules)