import sys
import os
import tempfile
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class CppBuildExt(build_ext):
    def build_extensions(self):

        if self.compiler.compiler_type == 'msvc':
            compile_args = ['/std:c++20', '/O2']
            link_args = []
        else:  # assuming GCC or Clang
            compile_args = ['-std:c++20', '-O3', '-Wno-unused-variable']
            link_args = []

        # Check for OpenMP support (enabled by default if available)
        if self._has_openmp_support():
            print("Compiler supports OpenMP. Building with parallelization enabled.")
            if self.compiler.compiler_type == 'msvc':
                compile_args.append('/openmp')
                link_args.append('/openmp')
            else:
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
        else:
            print("Compiler does not support OpenMP. Building in serial mode.")

        # Check for AVX2 support (opt-in via environment variable)
        if os.environ.get('PYTINYBVH_ENABLE_AVX2', '0') == '1':
            print("AVX2 support requested via environment variable.")
            if self._has_avx2_support():
                print("Compiler supports AVX2. Building with AVX2 optimizations.")
                if self.compiler.compiler_type == 'msvc':
                    compile_args.append('/arch:AVX2')
                else:
                    compile_args.extend(['-mavx2', '-mfma'])  # FMA is crucial for tinybvh's AVX2 paths
            else:
                print(
                    "WARNING: AVX2 support was requested, but the compiler does not support it. Building without AVX2.")
        else:
            print("Building without AVX2 optimizations. Set PYTINYBVH_ENABLE_AVX2=1 to enable.")

        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args

        super().build_extensions()

    def _compile_test_program(self, code, compile_args):
        """Helper to compile a small C++ program to test for feature support"""
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write(code)
            src_path = f.name

        obj_path = src_path.replace('.cpp', '.obj' if sys.platform == 'win32' else '.o')

        try:
            self.compiler.compile([src_path], extra_postargs=compile_args)
            return True
        except Exception:
            return False
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)
            if os.path.exists(obj_path):
                os.remove(obj_path)

    def _has_openmp_support(self):
        """Check if the compiler supports OpenMP"""
        cpp_code = """
        #include <omp.h>
        int main() {
            #pragma omp parallel
            { (void)omp_get_num_threads(); }
            return 0;
        }
        """
        return self._compile_test_program(
            cpp_code,
            ['/openmp'] if self.compiler.compiler_type == 'msvc' else ['-fopenmp']
        )

    def _has_avx2_support(self):
        """Check if the compiler supports AVX2"""
        cpp_code = """
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_setzero_ps();
            __m256 b = _mm256_setzero_ps();
            __m256 c = _mm256_add_ps(a, b);
            (void)c;
            return 0;
        }
        """
        return self._compile_test_program(
            cpp_code,
            ['/arch:AVX2'] if self.compiler.compiler_type == 'msvc' else ['-mavx2', '-mfma']
        )


ext_modules = [
    Extension(
        'pytinybvh',
        sources=['src/pytinybvh.cpp'],
        include_dirs=[
            pybind11.get_include(),
            'deps/',
        ],
        language='c++',
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': CppBuildExt},
)