import os
import sys
import tempfile
import platform
import textwrap
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

def try_compile(compiler, flags, code):
    """
    Try compiling a tiny translation unit with the given flags.
    Returns True on success, False on failure.
    """
    with tempfile.TemporaryDirectory() as td:
        src = Path(td, "test.cpp")
        src.write_text(code, encoding="utf-8")
        try:
            # MSVC uses /Fo for object output; unix compilers use -c -o
            if compiler.compiler_type == "msvc":
                # emulate minimal compile call
                cmd = compiler.compiler + flags + ["/c", str(src), "/Fo" + str(Path(td, "test.obj"))]
            else:
                cmd = compiler.compiler + flags + ["-c", str(src), "-o", str(Path(td, "test.o"))]
            compiler.spawn(cmd)
            return True
        except Exception:
            return False


def supports_avx2(compiler):
    # Needs immintrin and an AVX2 intrinsic to fail fast if not supported
    code = textwrap.dedent(
        r"""
        #include <immintrin.h>
        int main() {
            __m256i a = _mm256_set1_epi32(1);
            __m256i b = _mm256_set1_epi32(2);
            __m256i c = _mm256_add_epi32(a, b);
            (void)c;
            return 0;
        }
        """
    )
    if compiler.compiler_type == "msvc":
        return try_compile(compiler, ["/arch:AVX2"], code)
    else:
        # Clang/GCC usually also want -mfma when enabling AVX2 for math intrinsics
        return try_compile(compiler, ["-mavx2", "-mfma"], code)


def supports_avx(compiler):
    code = textwrap.dedent(
        r"""
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_set1_ps(1.0f);
            __m256 b = _mm256_set1_ps(2.0f);
            __m256 c = _mm256_add_ps(a, b);
            (void)c;
            return 0;
        }
        """
    )
    if compiler.compiler_type == "msvc":
        return try_compile(compiler, ["/arch:AVX"], code)
    else:
        return try_compile(compiler, ["-mavx"], code)

def is_x86_64():
    return platform.machine().lower() in ("x86_64", "amd64")

def is_arm64():
    return platform.machine().lower() in ("aarch64", "arm64")

def is_arm32():
    m = platform.machine().lower()
    return m.startswith("armv7") or m == "armv7l" or m == "armv6l" or m == "arm"

def supports_neon(compiler):
    # Simple NEON smoke test (works for armv7 and arm64)
    code = r"""
    #include <arm_neon.h>
    int main() {
        float32x4_t a = vdupq_n_f32(1.0f);
        float32x4_t b = vdupq_n_f32(2.0f);
        float32x4_t c = vaddq_f32(a, b);
        (void)c;
        return 0;
    }"""
    if compiler.compiler_type == "msvc":
        # On Windows ARM64, NEON is baseline, so we just try to compile without extra flags
        return try_compile(compiler, [], code)
    else:
        # For 32-bit ARM, compilers often require -mfpu=neon. On aarch64 it's baseline
        # So we try no flags first (covers aarch64), and then fall back to -mfpu=neon
        if try_compile(compiler, [], code):
            return True
        return try_compile(compiler, ["-mfpu=neon"], code)


class CppBuildExt(build_ext):
    def build_extensions(self):
        compile_args = []
        link_args = []

        # Baseline C++/warnings/opt flags
        if self.compiler.compiler_type == "msvc":
            compile_args += ["/std:c++20", "/O2", "/W3"]
        else:
            compile_args += ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Wno-unknown-pragmas"]

        # macOS + clang sometimes needs libc++ explicitly
        if sys.platform == "darwin" and self.compiler.compiler_type != "msvc":
            compile_args += ["-stdlib=libc++"]
            link_args += ["-stdlib=libc++"]

        # SIMD detection
        if os.environ.get("PYTINYBVH_NO_SIMD", "0") == "1":
            print("PYTINYBVH_NO_SIMD=1 is set. Building without AVX/AVX2.")

        elif is_x86_64():
            # Probe AVX2 first, then AVX
            if supports_avx2(self.compiler):
                if self.compiler.compiler_type == "msvc":
                    compile_args += ["/arch:AVX2"]
                else:
                    compile_args += ["-mavx2", "-mfma"]
                print("Enabling AVX2 (detected).")
            elif supports_avx(self.compiler):
                if self.compiler.compiler_type == "msvc":
                    compile_args += ["/arch:AVX"]
                else:
                    compile_args += ["-mavx"]
                print("Enabling AVX (detected).")
            else:
                print("No AVX/AVX2 support detected; building baseline.")
        else:
            print(f"Non-x86_64 machine ({platform.machine()}), skipping AVX flags.")

        # NEON detection
        if os.environ.get("PYTINYBVH_NO_SIMD", "0") == "1":
            print("PYTINYBVH_NO_SIMD=1 is set. Skipping NEON too.")

        elif is_arm64() or is_arm32():
            if supports_neon(self.compiler):
                if self.compiler.compiler_type == "msvc":
                    # No extra /arch flag needed for ARM64 MSVC. NEON is baseline.
                    print("NEON available (ARM64 baseline or MSVC ARM).")
                else:
                    # Only add -mfpu=neon if it was required by the test TU
                    # (if the no-flag compile succeeded, no need to add anything)

                    # Re-run quickly to see if flags were needed:
                    if not try_compile(self.compiler, [], "#include <arm_neon.h>\nint main(){return 0;}"):
                        compile_args += ["-mfpu=neon"]
                    print("Enabling NEON on ARM.")
            else:
                print("NEON not supported by this ARM target. Building scalar.")
        else:
            # Not an ARM machine; nothing to do.
            pass

        for ext in self.extensions:
            ext.extra_compile_args = list(getattr(ext, "extra_compile_args", [])) + compile_args
            ext.extra_link_args = list(getattr(ext, "extra_link_args", [])) + link_args

        super().build_extensions()



ext_modules = [
    Extension(
        "pytinybvh",
        sources=["src/pytinybvh.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "deps",
        ],
        language="c++",
    )
]

setup(
    name="pytinybvh",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CppBuildExt},
)
