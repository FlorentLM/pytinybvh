import sys
import os
import tempfile
import subprocess
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class CppBuildExt(build_ext):
    def build_extensions(self):

        # Determine compiler-specific flags
        if self.compiler.compiler_type == 'msvc':
            compile_args = ['/std:c++20', '/O2', '/W3']
            link_args = []
        else:  # GCC or Clang
            compile_args = ['-std=c++20', '-O3', '-Wno-unused-variable']
            link_args = []

        # Platform-specific flags for OpenMP and AVX
        current_platform = platform.system()
        current_machine = platform.machine()

        if self._has_openmp_support():
            print("Compiler supports OpenMP. Building with parallelization enabled.")
            if current_platform == "Windows":
                compile_args.append('/openmp')

            elif current_platform == "Darwin":
                compile_args.extend(['-Xpreprocessor', '-fopenmp'])
                link_args.append('-lomp')  # Link against Homebrew's libomp

            else:  # Linux
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
        else:
            print("Compiler does not support OpenMP. Building in serial mode.")

        # AVX
        if current_machine in ('x86_64', 'AMD64'):
            if self._has_avx2_support():
                print("Compiler supports AVX2. Building with AVX2 optimizations.")
                if current_platform == "Windows":
                    compile_args.append('/arch:AVX2')
                else:
                    compile_args.extend(['-mavx2', '-mfma'])
            elif self._has_avx_support():
                print("Compiler supports AVX. Building with AVX optimizations.")
                if current_platform == "Windows":
                    compile_args.append('/arch:AVX')
                else:
                    compile_args.append('-mavx')
        else:
            print(f"Machine is '{current_machine}', skipping AVX checks.")

        for ext in self.extensions:
            ext.extra_compile_args.extend(compile_args)
            ext.extra_link_args.extend(link_args)

            # Add Homebrew libomp paths if on macOS
            if current_platform == "Darwin":
                try:
                    brew_prefix = subprocess.check_output(['brew', '--prefix', 'libomp'], text=True).strip()
                    ext.include_dirs.append(os.path.join(brew_prefix, 'include'))
                    ext.library_dirs.append(os.path.join(brew_prefix, 'lib'))
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # This case is handled by _has_openmp_support, but we can pass here
                    pass

        super().build_extensions()

    def _compile_test_program(self, code, compile_args):
        """Helper to compile a small C++ program to test for feature support"""
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write(code)
            src_path = f.name

        obj_path = src_path.replace('.cpp', '.obj' if sys.platform == 'win32' else '.o')

        try:
            lib_dirs = []
            if platform.system() == "Darwin":
                try:
                    brew_prefix = subprocess.check_output(['brew', '--prefix', 'libomp'], text=True).strip()
                    lib_dirs.append(os.path.join(brew_prefix, 'lib'))
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

            self.compiler.compile([src_path], extra_postargs=compile_args, library_dirs=lib_dirs)
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
        if self.compiler.compiler_type == 'msvc':
            test_flags = ['/openmp']
        elif platform.system() == "Darwin":
            test_flags = ['-Xpreprocessor', '-fopenmp']
        else:  # Linux GCC/Clang
            test_flags = ['-fopenmp']

        return self._compile_test_program(cpp_code, test_flags)

    def _has_avx_support(self):
        """Check if the compiler supports AVX"""
        cpp_code = """
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_setzero_ps();
            (void)a;
            return 0;
        }
        """
        return self._compile_test_program(
            cpp_code,
            ['/arch:AVX'] if self.compiler.compiler_type == 'msvc' else ['-mavx']
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