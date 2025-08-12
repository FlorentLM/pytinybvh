#ifndef PYTINYBVH_CAPABILITIES_H
#define PYTINYBVH_CAPABILITIES_H

// -------------------------------- compile-time capabilities ---------------------------------

// This checks what SIMD instruction sets tinybvh is compiled with, using compiler flags (-msse4.2, /arch:AVX2, etc)

struct CompileTimeCapabilities {

    // Basic instruction set support, it's all based on tinybvh's preprocessor definitions
    const bool SSE4_2;
    const bool AVX;
    const bool AVX2;
    const bool NEON;

    // Traversal availability per BVH layout
    const bool SoA_trav;
    const bool BVH4CPU_trav;
    const bool CWBVH_trav;
    const bool BVH8CPU_trav;

    // The constructor is evaluated at compile time
    // (conversion between layouts is always possible, but native traversal requires the corresponding compiled code)

    constexpr CompileTimeCapabilities() :
        SSE4_2(
            #if defined(BVH_USESSE)
                true
            #else
                false
            #endif
        ),
        AVX(
            #if defined(BVH_USEAVX)
                true
            #else
                false
            #endif
        ),
        AVX2(
            #if defined(BVH_USEAVX2)
                true
            #else
                false
            #endif
        ),
        NEON(
            #if defined(BVH_USENEON)
                true
            #else
                false
            #endif
        ),
                                        //  In tinybvh:
        SoA_trav(AVX || NEON),          //    SoA traversal is optimized for AVX or NEON
        BVH4CPU_trav(SSE4_2 && !NEON),  //    BVH4CPU is specifically written using SSE intrinsics for x86
        CWBVH_trav(AVX),                //    Compressed Wide BVH requires AVX
        BVH8CPU_trav(AVX2)              //    BVH8 traversal is optimized for AVX2

    {}
};


// -------------------------------- runtime capabilities ---------------------------------

// This section contains functions to detect CPU capabilities at runtime
// Useful to detect mismatch between what's been compiled and what the hardware *actually* supports

// --------------------------------

// Platform-specific includes

#if defined(_MSC_VER)
    #include <intrin.h> // For __cpuidex (MSVC)
#elif defined(__x86_64__) || defined(__i386__)
    #include <cpuid.h>   // For __cpuid_count (GCC/Clang)
    #include <immintrin.h> // For _xgetbv intrinsic
#endif

#if (defined(__linux__) || defined(__ANDROID__)) && (defined(__aarch64__) || defined(__arm__))
    #include <sys/auxv.h> // for getauxval on Linux/Android (ARM)

    // Fallbacks for hwcap macros if the system headers are old
    #ifndef AT_HWCAP
        #define AT_HWCAP 16
    #endif

    #if defined(__aarch64__)
        #ifndef HWCAP_ASIMD
            #define HWCAP_ASIMD (1 << 1)
        #endif
    #else
        // 32-bit ARM
        #ifndef HWCAP_NEON
            #define HWCAP_NEON (1 << 12)
        #endif
    #endif

#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm__))
    #include <TargetConditionals.h>  // for Apple macros
#endif

// --------------------------------

// A simple struct to hold runtime CPU info
struct RuntimeCapabilities {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool neon = false;
};


// Some hieroglyphs below

/**
 * @brief A wrapper for the CPUID instruction on x86/x64.
 *
 * Provides info about the processor's features.
 *
 * @param leaf The main function leaf to query.
 * @param subleaf The sub-leaf (for extended functions).
 * @param regs Array of 4 integers to store the results (EAX, EBX, ECX, EDX).
 */
static void cpuid(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER)
    // use __cpuidex on MSVC
    __cpuidex(regs, leaf, subleaf);

#elif defined(__x86_64__) || defined(__i386__)
    // __cpuid_count is the equivalent of __cpuidex on GCC/Clang
    unsigned int a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    regs[0]=(int)a; regs[1]=(int)b; regs[2]=(int)c; regs[3]=(int)d;

#else
    // CPUID doesn't exist on non-x86 systems, so return zeros
    regs[0]=regs[1]=regs[2]=regs[3]=0;

#endif
}

/**
 * @brief Reads the XCR0 register on x86/x64.
 *
 * XCR0 (Extended Control Register 0) indicates which extended processor states the OS has enabled.
 * For AVX/AVX2, the OS must support saving and restoring YMM registers during context switches.
 *
 * @return The 64-bit value of the XCR0 register. Returns 0 on non-x86.
 */
static unsigned long long read_xcr0() {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(_MSC_VER)
    unsigned int eax, edx;

    // `xgetbv` reads an extended control register
    // (we use raw bytes because the `_xgetbv` intrinsic might not be available on all compilers.)
    // ECX=0 selects XCR0

    __asm__ volatile (".byte 0x0f, 0x01, 0xd0"
                      : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;

#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // `_xgetbv` should be available here
    return _xgetbv(0);

#else
    // if x86, return 0
    return 0;

#endif
}


/**
 * @brief Detects available SIMD instruction sets on the current CPU at runtime.
 *
 * @return A py::dict containing boolean flags for each instruction set:
 *         {"SSE4_2": bool, "AVX": bool, "AVX2": bool, "NEON": bool}
 */
static RuntimeCapabilities get_runtime_caps() {
    RuntimeCapabilities info;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

    // x86/x64 SIMD detection using CPUID

    int r[4];

    // Leaf 1: Basic CPU features
    cpuid(1, 0, r);
    bool has_sse42 = (r[2] & (1 << 20)) != 0;       // ECX bit 20: SSE4.2 support
    bool has_avx_cpu = (r[2] & (1 << 28)) != 0;     // ECX bit 28: AVX support
    bool os_uses_xsave = (r[2] & (1 << 27)) != 0;   // ECX bit 27: OSXSAVE flag

    bool ymm_state_enabled = false;
    if (os_uses_xsave && has_avx_cpu) {

        // The OSXSAVE flag bit means the OS supports saving extended states...
        // ...so, we read XCR0 to know *which* states it saves
        unsigned long long xcr0 = read_xcr0();

        // Check bits 1 (SSE) and 2 (YMM): both must be enabled for AVX
        // (0x6 corresponds to (1 << 1) | (1 << 2))
        ymm_state_enabled = (xcr0 & 0x6) == 0x6;
    }

    info.sse42 = has_sse42;
    info.avx = has_avx_cpu && ymm_state_enabled;

    // Leaf 7 sub-leaf 0: Extended features
    cpuid(7, 0, r);
    bool has_avx2_cpu = (r[1] & (1 << 5)) != 0; // EBX bit 5: AVX2 support

    // AVX2 also requires YMM state to be managed by the OS
    info.avx2 = has_avx2_cpu && ymm_state_enabled;

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // ARM/ARM64 NEON detection

    #if defined(__APPLE__)
        // For all Apple platforms (macOS on Apple Silicon, iOS, etc), NEON (or ASIMD)
        // is a baseline feature, so we can assume it's present
        #if TARGET_OS_IPHONE || TARGET_OS_OSX
            info.neon = true;
        #endif

    #elif defined(__linux__) || defined(__ANDROID__)
        // On Linux/Android, getauxval() is the usual way to query CPU features
        // (it reads the Auxiliary Vector provided by the kernel)
        unsigned long hw = getauxval(AT_HWCAP);

        #if defined(__aarch64__)
            // On AArch64, the feature is called ASIMD (Advanced SIMD)
            info.neon = (hw & HWCAP_ASIMD) != 0;
        #else
            // On 32-bit ARM, it's called NEON
            info.neon = (hw & HWCAP_NEON) != 0;
        #endif

    #elif defined(_WIN32)
        // For Windows on ARM64, NEON support is mandatory
        info.neon = true;
    #endif
#endif

    return info;
}

#endif //PYTINYBVH_CAPABILITIES_H