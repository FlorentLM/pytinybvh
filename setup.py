import os
import tempfile
import sys
from pathlib import Path
import platform
import textwrap
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

import pybind11


def _macos_arch():
    # 'arm64' on Apple Silicon, 'x86_64' on Intel
    return platform.machine()

def _sanitize_arch_flags(cmd):
    """Remove all -arch pairs. On macOS we will add back a single arch."""
    out = []
    skip = False
    for i, tok in enumerate(cmd):
        if skip:
            skip = False
            continue
        if tok == "-arch" and i + 1 < len(cmd):
            skip = True
            continue
        out.append(tok)
    return out



def try_compile(compiler, flags, code):
    """Tries to compile a given code snippet with a set of flags"""

    try:
        base_cmd = compiler.executables['compiler_cxx']
    except (KeyError, AttributeError):
        # Defensive fallback for bizarre environments. This should not be needed.
        if compiler.compiler_type == "msvc":
            base_cmd = ["cl.exe"]
        else:
            base_cmd = ["c++"] # or g++

    with tempfile.TemporaryDirectory() as td:
        src = Path(td, "test.cpp")
        src.write_text(code, encoding="utf-8")

        if compiler.compiler_type == "msvc":
            # On MSVC, base_cmd is ['cl.exe']. We just add flags
            cmd = base_cmd + flags + ["/c", str(src), "/Fo" + str(Path(td, "test.obj"))]
        else:
            # For GCC/Clang, we need a copy to modify for macOS
            current_base_cmd = list(base_cmd)
            if sys.platform == "darwin":
                current_base_cmd = _sanitize_arch_flags(current_base_cmd)
                current_base_cmd += ["-arch", _macos_arch()]

            cmd = current_base_cmd + flags + ["-c", str(src), "-o", str(Path(td, "test.o"))]

        try:
            compiler.spawn(cmd)
            return True

        except CompileError:
            return False

        except Exception:
            # for any other weird error
            return False

def supports_sse42(compiler):
    code = textwrap.dedent(
        r"""
        #include <nmmintrin.h> // For SSE4.2
        int main() {
            unsigned int crc = 0;
            crc = _mm_crc32_u8(crc, 1);
            (void)crc;
            return 0;
        }
        """
    )
    if compiler.compiler_type == "msvc":
        # On MSVC x64, SSE2 is baseline, and SSE4.2 is (generally) available
        # so no need for a /arch flag
        return try_compile(compiler, [], code)
    else:
        return try_compile(compiler, ["-msse4.2"], code)

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
    code = r"""
    #include <arm_neon.h>
    int main() {
        float32x4_t a = vdupq_n_f32(1.0f);
        float32x4_t b = vdupq_n_f32(2.0f);
        float32x4_t c = vaddq_f32(a, b);
        (void)c; return 0;
    }"""
    if is_arm64():
        # NEON is baseline on arm64, no-flag compile should work
        return try_compile(compiler, [], code)
    if is_arm32():
        # Some toolchains require -mfpu=neon (with matching -mfloat-abi set by the env)
        if try_compile(compiler, [], code):
            return True
        return try_compile(compiler, ["-mfpu=neon"], code)
    # Non-ARM
    return False


class CppBuildExt(build_ext):
    def build_extensions(self):
        compile_args = []
        link_args = []
        define_macros = []

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
            print("PYTINYBVH_NO_SIMD=1 is set. Building without SIMD.")

        elif is_x86_64():
            # Probe AVX2 first, then AVX, then SSE4.2
            if supports_avx2(self.compiler):
                if self.compiler.compiler_type == "msvc":
                    compile_args += ["/arch:AVX2"]
                    define_macros += [("__AVX__", "1"), ("__AVX2__", "1"), ("__FMA__", "1")]
                else:
                    compile_args += ["-mavx2", "-mfma"]
                print("Enabling AVX2 (detected).")
                define_macros += [("BVH_USEAVX2", "1")]

            elif supports_avx(self.compiler):
                if self.compiler.compiler_type == "msvc":
                    compile_args += ["/arch:AVX"]
                    define_macros += [("__AVX__", "1")]
                else:
                    compile_args += ["-mavx"]
                print("Enabling AVX (detected).")
                define_macros += [("BVH_USEAVX", "1")]

            elif supports_sse42(self.compiler):
                if self.compiler.compiler_type != "msvc":
                    compile_args += ["-msse4.2"]
                print("Enabling SSE4.2 (detected).")
                define_macros += [("BVH_USESSE", "1")]
            else:
                print("No AVX/SSE4.2 support detected. Building baseline.")

        # NEON detection
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
                    define_macros += [("BVH_USENEON", "1")]
            else:
                print("NEON not supported by this ARM target. Building baseline.")
        else:
            print(f"Unknown architecture ({platform.machine()}). Building baseline.")

        for ext in self.extensions:
            ext.extra_compile_args = list(getattr(ext, "extra_compile_args", [])) + compile_args
            ext.extra_link_args = list(getattr(ext, "extra_link_args", [])) + link_args
            ext.define_macros = list(getattr(ext, "define_macros", [])) + define_macros

        super().build_extensions()



ext_modules = [
    Extension(
        "pytinybvh",
        sources=["src/pytinybvh.cpp",],
        include_dirs=[
            pybind11.get_include(),
            "deps",
            "src",
        ],
        language="c++",
    )
]

setup(
    name="pytinybvh",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CppBuildExt},
)
