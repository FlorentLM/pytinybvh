cmake_minimum_required(VERSION 3.15)
project(pytinybvh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 REQUIRED
        COMPONENTS Interpreter ${DEV_MODULE} NumPy
        OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# Module target
nanobind_add_module(_pytinybvh src/pytinybvh.cpp)
set_target_properties(_pytinybvh PROPERTIES OUTPUT_NAME "_pytinybvh")

target_include_directories(_pytinybvh PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/deps"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Warnings & opt flags
if(MSVC)
    target_compile_options(_pytinybvh PRIVATE /W3 "$<$<CONFIG:Release>:/O2>")
else()
    target_compile_options(_pytinybvh PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_compile_options(_pytinybvh PRIVATE "$<$<CONFIG:Release>:-O3>" "$<$<CONFIG:Debug>:-g>")
    if(APPLE)
        target_compile_options(_pytinybvh PRIVATE -stdlib=libc++)
        target_link_libraries(_pytinybvh PRIVATE c++)
    endif()
endif()

# Feature macros
target_compile_definitions(_pytinybvh PRIVATE
        ENABLE_INDEXED_GEOMETRY
        ENABLE_CUSTOM_GEOMETRY
)

option(PYTINYBVH_NO_SIMD "Disable all SIMD optimizations (AVX, SSE, NEON)" OFF)

# SIMD detection
include(CheckCXXSourceCompiles)

if(PYTINYBVH_NO_SIMD)
    message(STATUS "PYTINYBVH_NO_SIMD is set. Building baseline without SIMD.")
else()
    # x86: AVX2 -> SSE4.2
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(CMAKE_REQUIRED_FLAGS "")
        set(AVX2_CODE "
        #include <immintrin.h>
        int main(){ __m256i a=_mm256_set1_epi32(1), b=_mm256_set1_epi32(2);
        auto c=_mm256_add_epi32(a,b); (void)c; return 0; }")
        if(MSVC)
            set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
        else()
            set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
        endif()
        check_cxx_source_compiles("${AVX2_CODE}" HAVE_AVX2)

        if(HAVE_AVX2)
            message(STATUS "Enabling AVX2 (detected).")
            target_compile_definitions(_pytinybvh PRIVATE BVH_USEAVX2)
            if(MSVC)
                target_compile_options(_pytinybvh PRIVATE /arch:AVX2)
            else()
                target_compile_options(_pytinybvh PRIVATE -mavx2 -mfma)
            endif()
        else()
            set(SSE42_CODE "
          #include <nmmintrin.h>
          int main(){ unsigned int crc=0; crc=_mm_crc32_u8(crc,1); return 0; }")
            if(MSVC)
                set(CMAKE_REQUIRED_FLAGS "")
            else()
                set(CMAKE_REQUIRED_FLAGS "-msse4.2")
            endif()
            check_cxx_source_compiles("${SSE42_CODE}" HAVE_SSE42)
            if(HAVE_SSE42)
                message(STATUS "Enabling SSE4.2 (detected).")
                target_compile_definitions(_pytinybvh PRIVATE BVH_USESSE)
                if(NOT MSVC)
                    target_compile_options(_pytinybvh PRIVATE -msse4.2)
                endif()
            else()
                message(STATUS "No AVX/SSE4.2 support detected. Building baseline.")
            endif()
        endif()
        set(CMAKE_REQUIRED_FLAGS "")
    endif()

    # ARM NEON
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
        set(CMAKE_REQUIRED_FLAGS "")
        set(NEON_CODE "
        #include <arm_neon.h>
        int main(){ auto a=vdupq_n_f32(1.0f), b=vdupq_n_f32(2.0f);
        auto c=vaddq_f32(a,b); (void)c; return 0; }")
        check_cxx_source_compiles("${NEON_CODE}" HAVE_NEON_BASELINE)
        if(HAVE_NEON_BASELINE)
            message(STATUS "NEON available (baseline).")
        else()
            set(CMAKE_REQUIRED_FLAGS "-mfpu=neon")
            check_cxx_source_compiles("${NEON_CODE}" HAVE_NEON_WITH_FLAG)
            if(HAVE_NEON_WITH_FLAG)
                message(STATUS "NEON available (enabled with -mfpu=neon).")
                target_compile_options(_pytinybvh PRIVATE -mfpu=neon)
            else()
                message(STATUS "NEON not supported. Building baseline.")
            endif()
        endif()
        set(CMAKE_REQUIRED_FLAGS "")
    endif()
endif()

# scikit-build-core + src/pytinybvh layout: drop the binary into the package dir
install(TARGETS _pytinybvh
        LIBRARY DESTINATION pytinybvh   # Unix/mac
        RUNTIME DESTINATION pytinybvh   # Windows .pyd
)

# Python stub file (.pyi) for IDE support and type checking
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pytinybvh.pyi"
        DESTINATION pytinybvh
)

# PEP 561 marker file to indicate the package is typed
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pytinybvh/py.typed"
        DESTINATION pytinybvh
)