#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/optional.h>

#define TINYBVH_IMPLEMENTATION
#include "tinybvh/tiny_bvh.h"

#include <vector>
#include <stdexcept>
#include <memory> // for std::unique_ptr
#include <string>
#include <cmath>
#include <cfloat>

#include "capabilities.h"

namespace nb = nanobind;
using namespace nb::literals;

constexpr float PI = 3.1415926535f;
constexpr float TWO_PI = (2.0f * PI);
constexpr float INV_PI = 1.0f / PI;
constexpr float INV_TWO_PI = 1.0f / TWO_PI;

// --------------------------- handy things to represent the library's capabilities ---------------------------

// Packet kernels require 64B aligned rays and an array stride a multiple of 64
constexpr bool TBVH_PACKET_SAFE = (alignof(tinybvh::Ray) >= 64) && ((sizeof(tinybvh::Ray) % 64) == 0);

// Single compile-time instance of the CPU info struct to use in helpers
constexpr CompileTimeCapabilities compiletime_caps;

// Public Layout enum (that one mirrors the one in tinybvh)
enum class Layout : uint32_t {
    Standard        = tinybvh::BVHBase::LAYOUT_BVH,
    SoA             = tinybvh::BVHBase::LAYOUT_BVH_SOA,
    BVH_GPU         = tinybvh::BVHBase::LAYOUT_BVH_GPU,
    BVH4_CPU        = tinybvh::BVHBase::LAYOUT_BVH4_CPU,
    BVH4_GPU        = tinybvh::BVHBase::LAYOUT_BVH4_GPU,
    CWBVH           = tinybvh::BVHBase::LAYOUT_CWBVH,
    BVH8_CPU        = tinybvh::BVHBase::LAYOUT_BVH8_AVX2,
    MBVH4           = 100, // custom value, tinybvh does not expose this
    MBVH8           = 101, // custom value, tinybvh does not expose this
};

// Helpers to convert layout enum to string

// For tinybvh layout
static inline const char* layout_to_string(tinybvh::BVHBase::BVHType L) {
    switch (L) {
        case tinybvh::BVHBase::LAYOUT_BVH:        return "Standard";
        case tinybvh::BVHBase::LAYOUT_BVH_SOA:    return "SoA";
        case tinybvh::BVHBase::LAYOUT_BVH_GPU:    return "BVH (GPU)";
        case tinybvh::BVHBase::LAYOUT_MBVH:       return "MBVH4";
        case tinybvh::BVHBase::LAYOUT_MBVH8:      return "MBVH8";
        case tinybvh::BVHBase::LAYOUT_BVH4_CPU:   return "BVH4 (CPU)";
        case tinybvh::BVHBase::LAYOUT_BVH4_GPU:   return "BVH4 (GPU)";
        case tinybvh::BVHBase::LAYOUT_CWBVH:      return "BVH8 (CWBVH)";
        case tinybvh::BVHBase::LAYOUT_BVH8_AVX2:  return "BVH8 (CPU)";
        default:                                  return "Unknown";
    }
}

// For our layout
static inline const char* layout_to_string(Layout L) {
    switch (L) {
        case Layout::Standard:                      return "Standard";
        case Layout::SoA:                           return "SoA";
        case Layout::BVH_GPU:                       return "BVH (GPU)";
        case Layout::MBVH4:                         return "MBVH4";
        case Layout::MBVH8:                         return "MBVH8";
        case Layout::BVH4_CPU:                      return "BVH4 (CPU)";
        case Layout::BVH4_GPU:                      return "BVH4 (GPU)";
        case Layout::CWBVH:                         return "BVH8 (CWBVH)";
        case Layout::BVH8_CPU:                      return "BVH8 (CPU)";
        default:                                    return "Unknown";
    }
}

static inline bool supports_layout(Layout L, bool for_traversal) {
    if (!for_traversal) return true; // conversion is always fine

    switch (L) {
        case Layout::Standard:  return true;
        case Layout::SoA:       return compiletime_caps.SoA_trav;
        case Layout::MBVH4:     return false; // NOLINT(*-branch-clone) Reason: being explicit is better here
        case Layout::MBVH8:     return false;
        case Layout::BVH4_CPU:  return compiletime_caps.BVH4CPU_trav;
        case Layout::CWBVH:     return compiletime_caps.CWBVH_trav;
        case Layout::BVH8_CPU:  return compiletime_caps.BVH8CPU_trav;
        case Layout::BVH_GPU:   return true; // NOLINT(*-branch-clone) Reason: being explicit is better here
        case Layout::BVH4_GPU:  return true;
    }
    return false;
}

static inline std::string explain_requirement(Layout L) {
    switch (L) {
        case Layout::SoA:      return "SoA traversal requires AVX or NEON.";
        case Layout::MBVH4:    return "MBVH layouts are for structural conversion, not direct traversal.";  // NOLINT(*-branch-clone) Reason: being explicit is better here
        case Layout::MBVH8:    return "MBVH layouts are for structural conversion, not direct traversal.";
        case Layout::BVH4_CPU: return "BVH4 (CPU) traversal requires SSE (x86).";
        case Layout::CWBVH:    return "BVH8 (CWBVH) traversal requires AVX.";
        case Layout::BVH8_CPU: return "BVH8 (CPU) traversal requires AVX2(+FMA).";
        case Layout::BVH_GPU:  return "BVH (GPU) traversal is scalar. No special ISA required.";
        case Layout::BVH4_GPU: return "BVH4 (GPU) traversal is scalar. No special ISA required.";
        case Layout::Standard: return "Standard layout always traverses.";
    }
    return "Unknown layout requirements.";
}

// -------------------------------- some other internal helpers and things ---------------------------------

// Cached imported modules (and intentionally "leak" them so no destructor runs during interpreter shutdown)
static nb::module_& np() { static auto* m = new nb::module_(nb::module_::import_("numpy")); return *m; }
static nb::module_& warnings() { static auto* w = new nb::module_(nb::module_::import_("warnings")); return *w; }
static nb::module_& userwarning() { static auto* u = new nb::module_(nb::module_::import_("UserWarning")); return *u; }
static nb::module_& builtins() { static auto* b = new nb::module_(nb::module_::import_("builtins")); return *b; }

// helper function to build the hardware info dict

nb::dict get_hardware_info() {
    nb::dict result, compile_time, runtime, compile_simd, runtime_simd, compile_layouts, rays;

    // Architecture
    #if defined(__aarch64__) || defined(_M_ARM64)
        result["architecture"] = "arm64";
    #elif defined(__arm__) || defined(_M_ARM)
        result["architecture"] = "arm";
    #elif defined(__x86_64__) || defined(_M_X64)
        result["architecture"] = "x86_64";
    #elif defined(__i386__) || defined(_M_IX86)
        result["architecture"] = "x86";
    #else
        result["architecture"] = "unknown";
    #endif

    // Compile-Time capabilities
    compile_simd["SSE4_2"] = compiletime_caps.SSE4_2;
    compile_simd["AVX"]    = compiletime_caps.AVX;
    compile_simd["AVX2"]   = compiletime_caps.AVX2;
    compile_simd["NEON"]   = compiletime_caps.NEON;
    compile_time["simd"]   = compile_simd;

    // mini helper to populate layout support info
    auto add_layout = [&](Layout L) {
        nb::dict row;
        row["convert"] = true; // conversion is always possible
        row["traverse"] = supports_layout(L, /*for_traversal=*/true);
        row["requirement"] = explain_requirement(L);
        compile_layouts[layout_to_string(L)] = row;
    };

    add_layout(Layout::Standard);
    add_layout(Layout::SoA);
    add_layout(Layout::BVH4_CPU);
    add_layout(Layout::CWBVH);
    add_layout(Layout::BVH8_CPU);
    add_layout(Layout::BVH_GPU);
    add_layout(Layout::BVH4_GPU);
    add_layout(Layout::MBVH4);
    add_layout(Layout::MBVH8);
    compile_time["layouts"] = compile_layouts;

    result["compile_time"] = compile_time;

    // Ray info
    rays["align"] = nb::int_(alignof(tinybvh::Ray));
    rays["size"]  = nb::int_(sizeof(tinybvh::Ray));
    rays["packet_safe"] = nb::bool_(TBVH_PACKET_SAFE);
    result["rays"] = rays;

    // Runtime capabilities
    RuntimeCapabilities rt_info = get_runtime_caps();
    runtime_simd["SSE4_2"] = rt_info.sse42;
    runtime_simd["AVX"]    = rt_info.avx;
    runtime_simd["AVX2"]   = rt_info.avx2;
    runtime_simd["NEON"]   = rt_info.neon;
    runtime["simd"] = runtime_simd;

    result["runtime"] = runtime;

    return result;
}

// Helper to fail fast if the binary was built with an ISA the current CPU/OS can't run
static inline void isa_compat_check() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    const RuntimeCapabilities rt = get_runtime_caps();
    if (compiletime_caps.AVX2 && !rt.avx2)
        throw std::runtime_error("pytinybvh was built with AVX2, but this CPU/OS doesn't support AVX2. Reinstall from source on this machine (no cached wheel), or set PYTINYBVH_NO_SIMD=1 for a baseline build.");
    if (compiletime_caps.AVX && !rt.avx)
        throw std::runtime_error("pytinybvh was built with AVX, but this CPU/OS doesn't support AVX. Reinstall from source on this machine (no cached wheel), or set PYTINYBVH_NO_SIMD=1 for a baseline build.");
    if (compiletime_caps.SSE4_2 && !rt.sse42)
        throw std::runtime_error("pytinybvh was built with SSE4.2, but this CPU does not support SSE4.2.");
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    RuntimeCapabilities rt = get_runtime_caps();
    if (compiletime_caps.NEON && !rt.neon)
        throw std::runtime_error("pytinybvh was built with NEON, but this device/OS does not expose NEON.");
#endif
}

// -------------------------------- internal structs ---------------------------------

// wrapper for tinybvh::Ray (to use in intersection queries)
struct PyRay {
    tinybvh::bvhvec3 origin;
    tinybvh::bvhvec3 direction;
    float t = 1e30f;
    float u = 0.0f, v = 0.0f;
    uint32_t prim_id = -1;
    uint32_t inst_id = -1;
    uint32_t mask = 0xFFFF;
};

struct HitRecord {
    uint32_t prim_id;
    uint32_t inst_id;
    float t, u, v;
};

struct TLASInstance {
    float transform[16];
    uint32_t blas_id;
    uint32_t mask;
};

// Context struct to hold pointers to geometry data for custom callbacks
struct CustomGeometryContext {
    // For AABBs
    const float* aabbs_ptr = nullptr;
    const float* inv_extents_ptr = nullptr;

    // For Spheres
    const float* points_ptr = nullptr;
    float sphere_radius = 0.0f;
};

// Custom dtypes
static PyObject* g_hit_record_dtype = nullptr;
static PyObject* g_bvh_node_dtype   = nullptr;
static PyObject* g_instance_dtype   = nullptr;

inline nb::object HITREC_DTYPE() { return nb::borrow(g_hit_record_dtype); }
inline nb::object BVHNODE_DTYPE() { return nb::borrow(g_bvh_node_dtype); }
inline nb::object INSTANCE_DTYPE(){ return nb::borrow(g_instance_dtype); }

// -------------------------------- these ones are exposed to python ---------------------------------

enum class BuildQuality { Quick, Balanced, High };
enum class GeometryType { Triangles, AABBs, Spheres };
enum class PacketMode { Auto, Never, Force };
enum class CachePolicy : uint8_t { ActiveOnly, All };

// ---------------------------------------------------------------------------------------------------


namespace {

    nb::object make_hit_record_dtype() {
        nb::dict spec;
        spec["names"]   = nb::make_tuple("prim_id", "inst_id", "t", "u", "v");
        spec["formats"] = nb::make_tuple(
            np().attr("uint32"), np().attr("uint32"),
            np().attr("float32"), np().attr("float32"), np().attr("float32")
        );
        spec["offsets"] = nb::make_tuple(
            nb::int_(offsetof(HitRecord, prim_id)),
            nb::int_(offsetof(HitRecord, inst_id)),
            nb::int_(offsetof(HitRecord, t)),
            nb::int_(offsetof(HitRecord, u)),
            nb::int_(offsetof(HitRecord, v))
        );
        spec["itemsize"] = nb::int_(sizeof(HitRecord));
        return np().attr("dtype")(spec);
    }

    nb::object make_bvh_node_dtype() {
        using Node = tinybvh::BVH::BVHNode;

        nb::dict spec;
        spec["names"]   = nb::make_tuple("aabb_min", "left_first", "aabb_max", "prim_count");
        // Subarray fields: (dtype, shape)
        spec["formats"] = nb::make_tuple(
            nb::make_tuple(np().attr("float32"), nb::make_tuple(3)),
            np().attr("uint32"),
            nb::make_tuple(np().attr("float32"), nb::make_tuple(3)),
            np().attr("uint32")
        );
        spec["offsets"] = nb::make_tuple(
            nb::int_(offsetof(Node, aabbMin)),
            nb::int_(offsetof(Node, leftFirst)),
            nb::int_(offsetof(Node, aabbMax)),
            nb::int_(offsetof(Node, triCount))
        );
        spec["itemsize"] = nb::int_(sizeof(Node));
        return np().attr("dtype")(spec);
    }

    nb::object make_instance_dtype() {
        nb::dict spec;

        spec["names"]   = nb::make_tuple("transform", "blas_id", "mask");
        spec["formats"] = nb::make_tuple(
            nb::make_tuple(np().attr("float32"), nb::make_tuple(4, 4)),
            np().attr("uint32"),
            np().attr("uint32")
        );
        spec["offsets"] = nb::make_tuple(
            nb::int_(offsetof(TLASInstance, transform)),
            nb::int_(offsetof(TLASInstance, blas_id)),
            nb::int_(offsetof(TLASInstance, mask))
        );
        spec["itemsize"] = nb::int_(sizeof(TLASInstance));
        return np().attr("dtype")(spec);
    }
}

// Small default context for building a BVH
static inline tinybvh::BVHContext default_ctx() {
    tinybvh::BVHContext ctx{};
    ctx.malloc = tinybvh::malloc64;
    ctx.free   = tinybvh::free64;
    ctx.userdata = nullptr;
    return ctx;
}

// BVH4_CPU and BVH8_CPU need a matching allocator/free pair
// Reason: BVH4_CPU and BVH8_CPU allocate with `malloc4k` when tinybvh is compiled with `BVH8_ALIGN_4K` (the default)
// and the destructor uses context.free so if we used the default context, it would use free64 and cause heap mismatch
static inline tinybvh::BVHContext cpu_wide_ctx() {
    tinybvh::BVHContext ctx{};
#if defined(BVH8_ALIGN_32K)
    ctx.malloc = tinybvh::malloc32k;
    ctx.free   = tinybvh::free32k;
#elif defined(BVH8_ALIGN_4K)
    ctx.malloc = tinybvh::malloc4k;
    ctx.free   = tinybvh::free4k;
#else
    ctx.malloc = tinybvh::malloc64;
    ctx.free   = tinybvh::free64;
#endif
    ctx.userdata = nullptr;
    return ctx;
}

// type caster to accept tinybvh::bvhvec3 objects
template <> struct nanobind::detail::type_caster<tinybvh::bvhvec3> {
    NB_TYPE_CASTER(tinybvh::bvhvec3, const_name("bvhvec3"));

    // Python -> C++
    bool from_python(nb::handle src, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        nb::object obj = nb::borrow(src);

        // Fast path: (3, ) float32 contiguous
        nb::ndarray<const float,  nb::shape<3>, nb::c_contig> a32;
        if (nb::try_cast(obj, a32)) {
            value = { a32(0), a32(1), a32(2) };
            return true;
        }
        // Fast path: (3, ) float64 contiguous
        nb::ndarray<const double, nb::shape<3>, nb::c_contig> a64;
        if (nb::try_cast(obj, a64)) {
            value = { static_cast<float>(a64(0)), static_cast<float>(a64(1)), static_cast<float>(a64(2)) };
            return true;
        }
        // 1D ndarray of length 3: index via sequence
        nb::ndarray<> any;
        if (nb::try_cast(obj, any) && any.ndim() == 1 && any.size() == 3) {
            nb::sequence s;
            if (!nb::try_cast(obj, s)) return false;
            float x, y, z;
            if (!nb::try_cast(s[0], x) || !nb::try_cast(s[1], y) || !nb::try_cast(s[2], z))
                return false;
            value = { x, y, z };
            return true;
        }
        // Generic 3-element sequence (list, tuple, etc)
        nb::sequence s;
        if (nb::try_cast(obj, s) && nb::len(s) == 3) {
            float x, y, z;
            if (!nb::try_cast(s[0], x) || !nb::try_cast(s[1], y) || !nb::try_cast(s[2], z))
                return false;
            value = { x, y, z };
            return true;
        }
        return false;
    }

    // C++ -> Python
    static nb::handle from_cpp(const tinybvh::bvhvec3 &v, nb::rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        // Allocate owned memory and wrap as ndarray -> Python via .cast()
        auto *data = new (std::nothrow) float[3]{ v.x, v.y, v.z };
        if (!data) return {};
        nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<float *>(p); });

        using Vec3 = nb::ndarray<float, nb::numpy, nb::shape<3>, nb::c_contig>;
        Vec3 arr(data, { 3 }, owner);
        return arr.cast().release();  // materialize numpy array
    }
}; // namespace nanobind::detail


// This pointer is only for the duration of the BVH build process for AABBs (and points)
static const float* g_build_aabbs_ptr = nullptr;

struct AABBptrGuard {
    explicit AABBptrGuard(const float* p) {
        if (g_build_aabbs_ptr) throw std::runtime_error("Nested AABB builds not supported.");
        g_build_aabbs_ptr = p;
    }
    ~AABBptrGuard() { g_build_aabbs_ptr = nullptr; }
};

// C-style callback for building from AABBs
static void aabb_build_callback(unsigned int i, tinybvh::bvhvec3& bmin, tinybvh::bvhvec3& bmax) {
    const float* a = g_build_aabbs_ptr;
    const size_t off = static_cast<size_t>(i) * 6;
    bmin.x = a[off+0]; bmin.y = a[off+1]; bmin.z = a[off+2];
    bmax.x = a[off+3]; bmax.y = a[off+4]; bmax.z = a[off+5];
}

// C-style callback for intersecting AABBs
bool aabb_intersect_callback(tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!ray.hit.auxData) throw std::runtime_error("Internal error: Ray context is null in AABB intersect callback.");
    auto* context = static_cast<const CustomGeometryContext*>(ray.hit.auxData);

    const size_t offset = prim_id * 6;
    const tinybvh::bvhvec3 bmin = {context->aabbs_ptr[offset], context->aabbs_ptr[offset + 1], context->aabbs_ptr[offset + 2]};
    const tinybvh::bvhvec3 bmax = {context->aabbs_ptr[offset + 3], context->aabbs_ptr[offset + 4], context->aabbs_ptr[offset + 5]};

    float t = tinybvh::tinybvh_intersect_aabb(ray, bmin, bmax);

    if (t < ray.hit.t) {
        ray.hit.t = t;
        ray.hit.prim = prim_id;

        // Calculate UV coordinates on the hit face
        const tinybvh::bvhvec3 hit_point = ray.O + ray.D * t;

        // Use pre-computed reciprocal extent from the context
        const size_t inv_offset = prim_id * 3;
        const tinybvh::bvhvec3 inv_extent = {context->inv_extents_ptr[inv_offset], context->inv_extents_ptr[inv_offset + 1], context->inv_extents_ptr[inv_offset + 2]};

        const float dx0 = std::abs(hit_point.x - bmin.x);
        const float dx1 = std::abs(hit_point.x - bmax.x);
        const float dy0 = std::abs(hit_point.y - bmin.y);
        const float dy1 = std::abs(hit_point.y - bmax.y);
        const float dz0 = std::abs(hit_point.z - bmin.z);
        const float dz1 = std::abs(hit_point.z - bmax.z);

        float min_dist = dx0;
        int face_id = 0; // -X face

        if (dx1 < min_dist) { min_dist = dx1; face_id = 1; }
        if (dy0 < min_dist) { min_dist = dy0; face_id = 2; }
        if (dy1 < min_dist) { min_dist = dy1; face_id = 3; }
        if (dz0 < min_dist) { min_dist = dz0; face_id = 4; }
        if (dz1 < min_dist) { min_dist = dz1; face_id = 5; }

        switch (face_id) {
            case 0: // -X face
                ray.hit.u = (hit_point.z - bmin.z) * inv_extent.z;
                ray.hit.v = (bmax.y - hit_point.y) * inv_extent.y;
                break;
            case 1: // +X face
                ray.hit.u = (bmax.z - hit_point.z) * inv_extent.z; // Flipped u for consistency
                ray.hit.v = (bmax.y - hit_point.y) * inv_extent.y; // Flipped v for consistency
                break;
            case 2: // -Y face
                ray.hit.u = (hit_point.x - bmin.x) * inv_extent.x;
                ray.hit.v = (hit_point.z - bmin.z) * inv_extent.z;
                break;
            case 3: // +Y face
                ray.hit.u = (hit_point.x - bmin.x) * inv_extent.x;
                ray.hit.v = (bmax.z - hit_point.z) * inv_extent.z; // Flipped v for consistency
                break;
            case 4: // -Z face
                ray.hit.u = (bmax.x - hit_point.x) * inv_extent.x; // Flipped u for consistency
                ray.hit.v = (bmax.y - hit_point.y) * inv_extent.y; // Flipped v for consistency
                break;
            case 5: // +Z face
                ray.hit.u = (hit_point.x - bmin.x) * inv_extent.x;
                ray.hit.v = (bmax.y - hit_point.y) * inv_extent.y; // Flipped v for consistency
                break;
            default: ;
        }

        return true;
    }
    return false;
}

// C-style callback for occlusion testing with AABBs
bool aabb_isoccluded_callback(const tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!ray.hit.auxData) throw std::runtime_error("Internal error: Ray context is null in AABB occlusion callback.");
    auto* context = static_cast<const CustomGeometryContext*>(ray.hit.auxData);

    const size_t offset = prim_id * 6;
    const tinybvh::bvhvec3 bmin = {context->aabbs_ptr[offset], context->aabbs_ptr[offset + 1], context->aabbs_ptr[offset + 2]};
    const tinybvh::bvhvec3 bmax = {context->aabbs_ptr[offset + 3], context->aabbs_ptr[offset + 4], context->aabbs_ptr[offset + 5]};

    // slab test
    float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	float tmin = tinybvh::tinybvh_min( tx1, tx2 ), tmax = tinybvh::tinybvh_max( tx1, tx2 );
	float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;

	tmin = tinybvh::tinybvh_max( tmin, tinybvh::tinybvh_min( ty1, ty2 ) );
	tmax = tinybvh::tinybvh_min( tmax, tinybvh::tinybvh_max( ty1, ty2 ) );

	float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;

	tmin = tinybvh::tinybvh_max( tmin, tinybvh::tinybvh_min( tz1, tz2 ) );
	tmax = tinybvh::tinybvh_min( tmax, tinybvh::tinybvh_max( tz1, tz2 ) );

    return (tmax >= tmin && tmin < ray.hit.t && tmax >= 0);
}

// Ray-sphere intersection helper: returns hit distance `t` if an intersection is found within ray.hit.t
std::optional<float> intersect_sphere_primitive(const tinybvh::Ray& ray, const tinybvh::bvhvec3& center, float radius_sq) {
    const tinybvh::bvhvec3 oc = ray.O - center;
    const float b = tinybvh_dot(oc, ray.D); // a=1 for normalized ray.D
    const float c = tinybvh_dot(oc, oc) - radius_sq;
    const float discriminant = b * b - c; // a=1.0

    if (discriminant < 0) return std::nullopt;

    float t = -b - sqrt(discriminant);
    if (t > 1e-6f && t < ray.hit.t) {
        return t;
    }
    return std::nullopt;
}

// C-style callback for ray-sphere intersection
bool sphere_intersect_callback(tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!ray.hit.auxData) throw std::runtime_error("Internal error: Ray context is null in sphere intersect callback.");
    auto* context = static_cast<const CustomGeometryContext*>(ray.hit.auxData);

    const size_t offset = prim_id * 3;
    const tinybvh::bvhvec3 center = {context->points_ptr[offset], context->points_ptr[offset + 1], context->points_ptr[offset + 2]};
    const float radius_sq = context->sphere_radius * context->sphere_radius;

    if (auto t = intersect_sphere_primitive(ray, center, radius_sq)) {
        ray.hit.t = *t;
        ray.hit.prim = prim_id;

        const tinybvh::bvhvec3 hit_point = ray.O + ray.D * (*t);
        const tinybvh::bvhvec3 normal = tinybvh::tinybvh_normalize(hit_point - center);

        ray.hit.u = 0.5f + atan2f(normal.z, normal.x) * INV_TWO_PI;
        ray.hit.v = 0.5f - asinf(normal.y) * INV_PI;
        return true;
    }
    return false;
}

// C-style callback for occlusion testing with spheres
bool sphere_isoccluded_callback(const tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!ray.hit.auxData) throw std::runtime_error("Internal error: Ray context is null in sphere occlusion callback.");
    auto* context = static_cast<const CustomGeometryContext*>(ray.hit.auxData);

    const size_t offset = prim_id * 3;
    const tinybvh::bvhvec3 center = {context->points_ptr[offset], context->points_ptr[offset + 1], context->points_ptr[offset + 2]};
    const float radius_sq = context->sphere_radius * context->sphere_radius;

    return intersect_sphere_primitive(ray, center, radius_sq).has_value();
}


// Packet gating helpers (TU-local), for batch intersect and batch occlusion tests

namespace {

    // Check all origins are the same within eps
    inline bool same_origin_ok(const float* origins, size_t o_row, size_t o_col,
                               size_t n, float eps)
    {
        if (n <= 1) return true;
        const float ox0 = origins[0 * o_col];
        const float oy0 = origins[1 * o_col];
        const float oz0 = origins[2 * o_col];
        for (size_t i = 1; i < n; ++i) {
            const size_t base = i * o_row;
            const float dx = std::fabs(origins[base + 0 * o_col] - ox0);
            const float dy = std::fabs(origins[base + 1 * o_col] - oy0);
            const float dz = std::fabs(origins[base + 2 * o_col] - oz0);
            if (dx > eps || dy > eps || dz > eps) return false;
        }
        return true;
    }

    // Per-block cone/tmax tests
    struct PacketGate {
        const float* dirs;        // (N,3)
        size_t       d_row, d_col;
        const float* tmax;        // nullable
        float        cos_thresh;  // cos(max_cone_deg)
        float        tmax_ratio;  // allowed tmax band

        [[nodiscard]] inline bool cone_ok(Py_ssize_t start, int count) const {
            float mx = 0.f, my = 0.f, mz = 0.f;
            for (int j = 0; j < count; ++j) {
                const size_t base = static_cast<size_t>(start + j) * d_row;
                mx += dirs[base + 0 * d_col];
                my += dirs[base + 1 * d_col];
                mz += dirs[base + 2 * d_col];
            }
            const float len = std::sqrt(mx*mx + my*my + mz*mz);
            if (len < 1e-20f) return false;
            const float ux = mx / len, uy = my / len, uz = mz / len;

            float min_dot = 1.0f;
            for (int j = 0; j < count; ++j) {
                const size_t base = static_cast<size_t>(start + j) * d_row;
                const float dx = dirs[base + 0 * d_col];
                const float dy = dirs[base + 1 * d_col];
                const float dz = dirs[base + 2 * d_col];
                const float dot = dx*ux + dy*uy + dz*uz;
                if (dot < min_dot) {
                    min_dot = dot;
                    if (min_dot < cos_thresh) return false; // early out
                }
            }
            return true;
        }

        static inline bool tmax_ok(Py_ssize_t start, int count) {
        // [[nodiscard]] inline bool tmax_ok(Py_ssize_t start, int count) {
            // if (!tmax) return true;
            // float tmin = FLT_MAX, tmaxv = 0.f;
            // for (int j = 0; j < count; ++j) {
            //     const float t = tmax[start + j];
            //     if (t < tmin) tmin = t;
            //     if (t > tmaxv) tmaxv = t;
            // }
            // tmin = std::max(1e-6f, tmin);
            // return (tmaxv <= tmin * tmax_ratio);
            return true;
        }
    };

} // anonymous namespace


// -------------------------------- main pytinybvh classes ---------------------------------

// Helper to dispatch calls to the correct BVH implementation based on its layout
// (the static_cast is safe because we check the 'layout' enum before casting)
template <typename Func>
auto dispatch_bvh_call(tinybvh::BVHBase* bvh_base, Func f) {
    if (!bvh_base) {
        throw std::runtime_error("Cannot dispatch call on a null BVH.");
    }
    // NOLINTBEGIN(cppcoreguidelines-pro-type-static-cast-downcast) // Reason: see above
    switch (bvh_base->layout) {
        case tinybvh::BVHBase::LAYOUT_BVH:
            return f(*static_cast<tinybvh::BVH*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_BVH_SOA:
            return f(*static_cast<tinybvh::BVH_SoA*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_BVH_GPU:
            return f(*static_cast<tinybvh::BVH_GPU*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_BVH4_CPU:
            return f(*static_cast<tinybvh::BVH4_CPU*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_BVH4_GPU:
            return f(*static_cast<tinybvh::BVH4_GPU*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_CWBVH:
            return f(*static_cast<tinybvh::BVH8_CWBVH*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_BVH8_AVX2:
            return f(*static_cast<tinybvh::BVH8_CPU*>(bvh_base));
        case tinybvh::BVHBase::UNDEFINED:
            throw std::runtime_error("Empty/undefined BVH.");
        case tinybvh::BVHBase::LAYOUT_MBVH:
            throw std::runtime_error(
            "Operation is not supported for MBVH layouts directly. "
            "Convert to a traversable layout like Standard, SoA, or BVH4_CPU first.");
        default:
            throw std::runtime_error("Operation is not supported for this BVH layout.");
    }
    // NOLINTEND(cppcoreguidelines-pro-type-static-cast-downcast)
}

// C++ class to wrap the tinybvh::BVH object and its associated data: this is the object held by the Python BVH class
struct PyBVH {

    // The canonical BVH in standard layout, always present after build
    std::unique_ptr<tinybvh::BVH> base_;

    // optional converted layouts for caching
    std::unique_ptr<tinybvh::BVH_SoA>     soa_;
    std::unique_ptr<tinybvh::MBVH<4>>     mbvh4_;
    std::unique_ptr<tinybvh::MBVH<8>>     mbvh8_;
    std::unique_ptr<tinybvh::BVH4_CPU>    bvh4_cpu_;
    std::unique_ptr<tinybvh::BVH4_GPU>    bvh4_gpu_;
    std::unique_ptr<tinybvh::BVH8_CWBVH>  cwbvh_;
    std::unique_ptr<tinybvh::BVH8_CPU>    bvh8_cpu_;
    std::unique_ptr<tinybvh::BVH_GPU>     bvh_gpu_;

    // A raw pointer to the currently active BVH representation
    // This is not an owning pointer, it points to one of the objects above
    tinybvh::BVHBase* active_bvh_ = nullptr;
    Layout active_layout_ = Layout::Standard;
    CachePolicy cache_policy_ = CachePolicy::ActiveOnly;

    // These references must be kept alive for the lifetime of any BVH representation
    nb::list source_geometry_refs;
    std::vector<tinybvh::BLASInstance> instances_data;
    nb::ndarray<uint32_t, nb::c_contig> opacity_map_ref;  // ref to keep opacity maps
    std::vector<tinybvh::BVHBase*> blas_pointers; // raw pointers to BLASes for TLAS build

    // Metadata about the BVH that persists across conversions
    BuildQuality quality = BuildQuality::Balanced;
    enum class CustomType { None, AABB, Sphere };
    CustomType custom_type = CustomType::None;
    float sphere_radius = 0.0f;

    // Constructor
    PyBVH() = default;

private:
    void ensure_mbvh4(bool compact) {
        if (!mbvh4_) mbvh4_ = std::make_unique<tinybvh::MBVH<4>>();
        // Make sure the allocator is valid and current
        if (!mbvh4_->context.malloc) mbvh4_->context = default_ctx();
        mbvh4_->context = base_->context;
        mbvh4_->ConvertFrom(*base_, compact);   // always refresh
    }

    void ensure_mbvh8(bool compact) {
        if (!mbvh8_) mbvh8_ = std::make_unique<tinybvh::MBVH<8>>();
        if (!mbvh8_->context.malloc) mbvh8_->context = default_ctx();
        mbvh8_->context = base_->context;
        mbvh8_->ConvertFrom(*base_, compact);   // always refresh
    }

public:

    // Cache policy management

    void set_cache_policy(CachePolicy p) { cache_policy_ = p; }

    void clear_cached_layouts() {
        const bool needs_m4 =
            (active_layout_ == Layout::MBVH4) ||
            (active_layout_ == Layout::BVH4_CPU) ||
            (active_layout_ == Layout::BVH4_GPU);

        const bool needs_m8 =
            (active_layout_ == Layout::MBVH8) ||
            (active_layout_ == Layout::CWBVH)  ||
            (active_layout_ == Layout::BVH8_CPU);

        if (active_layout_ != Layout::BVH4_CPU) {
            // have to make make sure the wide CPU layouts use the wide context so their destructor frees with the right function
            if (bvh4_cpu_) bvh4_cpu_->context = cpu_wide_ctx();
            bvh4_cpu_.reset();
        }

        if (active_layout_ != Layout::BVH8_CPU) {
            // have to make sure the wide CPU layouts use the wide context so their destructor frees with the right function
            if (bvh8_cpu_) bvh8_cpu_->context = cpu_wide_ctx();
            bvh8_cpu_.reset();
        }

        if (active_layout_ != Layout::SoA)      soa_.reset();

        // Only drop MBVH if nothing currently depends on it
        if (!needs_m4)                           mbvh4_.reset();
        if (!needs_m8)                           mbvh8_.reset();

        if (active_layout_ != Layout::BVH4_GPU)  bvh4_gpu_.reset();
        if (active_layout_ != Layout::CWBVH)     cwbvh_.reset();
        if (active_layout_ != Layout::BVH_GPU)   bvh_gpu_.reset();
    }

    // Converter method

    void convert_to(Layout target, bool compact=true, bool strict=false) {
        if (!base_) {
            throw std::runtime_error("Cannot convert an uninitialized BVH.");
        }

        auto has_alloc = [](const tinybvh::BVHBase& b){ return b.context.malloc && b.context.free; };
        if (!has_alloc(*base_)) {
            throw std::runtime_error("BVH context has no aligned allocator. Initialize context before conversion.");
        }

        if (active_layout_ == target && active_bvh_) {
            // already in desired layout, nothing to do
            return;
        }

        // Always allows conversion. Enforces traversal availability only if `strict` is True
        if (strict && !supports_layout(target, /*for_traversal=*/true)) {
            throw std::runtime_error(
                std::string("Conversion to ") + layout_to_string(target) + ": " +
                explain_requirement(target));
        }
        switch (target) {
            case Layout::Standard: {
                active_bvh_ = base_.get();
                active_bvh_->context = default_ctx();
                active_layout_ = Layout::Standard;
                break;
            }
            case Layout::SoA: {
                if (!soa_) soa_ = std::make_unique<tinybvh::BVH_SoA>();
                soa_->context = base_->context;
                soa_->ConvertFrom(*base_, compact);
                active_bvh_ = soa_.get();
                active_layout_ = Layout::SoA;
                break;
            }
            case Layout::BVH_GPU: {
                if (!bvh_gpu_) bvh_gpu_ = std::make_unique<tinybvh::BVH_GPU>();
                bvh_gpu_->context = base_->context;
                bvh_gpu_->ConvertFrom(*base_, compact);
                active_bvh_ = bvh_gpu_.get();
                active_layout_ = Layout::BVH_GPU;
                break;
            }
            case Layout::MBVH4: {
                ensure_mbvh4(compact);
                active_bvh_ = mbvh4_.get();
                active_layout_ = Layout::MBVH4;
                break;
            }
            case Layout::MBVH8: {
                ensure_mbvh8(compact);
                active_bvh_ = mbvh8_.get();
                active_layout_ = Layout::MBVH8;
                break;
            }
            case Layout::BVH4_CPU: {
                if (!mbvh4_) mbvh4_ = std::make_unique<tinybvh::MBVH<4>>();
                mbvh4_->context = base_->context;
                mbvh4_->ConvertFrom(*base_, compact);

                if (!bvh4_cpu_) bvh4_cpu_ = std::make_unique<tinybvh::BVH4_CPU>();
                bvh4_cpu_->context = cpu_wide_ctx();    // TODO: Why doesn't that solve the crash????
                bvh4_cpu_->ConvertFrom(*mbvh4_);
                bvh4_cpu_->context = cpu_wide_ctx();
                active_bvh_ = bvh4_cpu_.get();
                active_layout_ = Layout::BVH4_CPU;
                break;
            }
            case Layout::BVH4_GPU: {
                ensure_mbvh4(compact);
                if (!bvh4_gpu_) bvh4_gpu_ = std::make_unique<tinybvh::BVH4_GPU>();
                bvh4_gpu_->context = base_->context;
                bvh4_gpu_->ConvertFrom(*mbvh4_, compact);
                active_bvh_ = bvh4_gpu_.get();
                active_layout_ = Layout::BVH4_GPU;
                break;
            }
            case Layout::CWBVH: {
                ensure_mbvh8(compact);
                if (!cwbvh_) cwbvh_ = std::make_unique<tinybvh::BVH8_CWBVH>();
                cwbvh_->context = base_->context;
                cwbvh_->ConvertFrom(*mbvh8_, compact);
                active_bvh_ = cwbvh_.get();
                active_layout_ = Layout::CWBVH;
                break;
            }
            case Layout::BVH8_CPU: {
                if (!mbvh8_) mbvh8_ = std::make_unique<tinybvh::MBVH<8>>();
                mbvh8_->context = base_->context;
                mbvh8_->ConvertFrom(*base_, compact);

                if (!bvh8_cpu_) bvh8_cpu_ = std::make_unique<tinybvh::BVH8_CPU>();
                bvh8_cpu_->context = cpu_wide_ctx();
                bvh8_cpu_->ConvertFrom(*mbvh8_);
                bvh8_cpu_->context = cpu_wide_ctx();
                active_bvh_ = bvh8_cpu_.get();
                active_layout_ = Layout::BVH8_CPU;
                break;
            }

        }
        if (cache_policy_ == CachePolicy::ActiveOnly) {
            clear_cached_layouts();
        }
    }

    // Helper for builders to finalize the wrapper
    static void finalize_build(const std::unique_ptr<PyBVH>& wrapper, std::unique_ptr<tinybvh::BVH> bvh) {
        bvh->context = default_ctx();
        wrapper->base_ = std::move(bvh);
        wrapper->active_bvh_ = wrapper->base_.get();
        wrapper->active_layout_ = Layout::Standard;
    }

    // Core builders (zero-copy)

    static std::unique_ptr<PyBVH> from_vertices(
        nb::ndarray<const float, nb::c_contig> vertices_np,
        const BuildQuality quality,
        const float traversal_cost,
        const float intersection_cost,
        const int hq_bins = HQBVHBINS) {

        if (vertices_np.dtype() != nb::dtype<float>()) {
            throw nb::type_error("Input `vertices` must be a float32 numpy array.");
        }
        if (vertices_np.ndim() != 2 || vertices_np.shape(0) % 3 != 0 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input `vertices` must be a 2D numpy array with shape (N*3, 4)");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->source_geometry_refs.append(vertices_np);   // reference to the numpy array
        wrapper->quality = quality;
        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;
        bvh->hqbvhbins = hq_bins;

        if (vertices_np.shape(0) == 0) {
            finalize_build(wrapper, std::move(bvh));
            return wrapper;
        }

        auto vertices_ptr = reinterpret_cast<const tinybvh::bvhvec4*>(vertices_np.data());
        const auto prim_count = static_cast<uint32_t>(vertices_np.shape(0) / 3);

        bvh->context = default_ctx();
        {
            nb::gil_scoped_release release;
            switch (quality) {
                case BuildQuality::Quick:
                    bvh->BuildQuick(vertices_ptr, prim_count);
                    break;
                case BuildQuality::High:
                    bvh->BuildHQ(vertices_ptr, prim_count);
                    break;
                case BuildQuality::Balanced:
                default:
                    bvh->Build(vertices_ptr, prim_count);
                    break;
            }
        }
        finalize_build(wrapper, std::move(bvh)); // transfer ownership
        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_indexed_mesh(
        nb::ndarray<const float, nb::c_contig> vertices_np,
        nb::ndarray<const uint32_t, nb::c_contig> indices_np,
        const BuildQuality quality,
        const float traversal_cost,
        const float intersection_cost,
        const int hq_bins = HQBVHBINS) {

        if (vertices_np.dtype() != nb::dtype<float>()) {
            throw nb::type_error("Input `vertices` must be a float32 numpy array.");
        }
        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input `vertices` must be a 2D numpy array with shape (N, 4)");
        }
        if (indices_np.dtype() != nb::dtype<uint32_t>()) {
            throw nb::type_error("Input `indices` must be a uint32 numpy array.");
        }
        if (indices_np.ndim() != 2 || indices_np.shape(1) != 3) {
            throw std::runtime_error("Input `indices` must be a 2D numpy array with shape (N, 3)");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->source_geometry_refs.append(vertices_np);  // references to vertices numpy array
        wrapper->source_geometry_refs.append(indices_np);   // and indexes numpy array
        wrapper->quality = quality;

        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;
        bvh->hqbvhbins = hq_bins;

        if (vertices_np.shape(0) == 0  || indices_np.shape(0) == 0) {
            finalize_build(wrapper, std::move(bvh));  // transfer ownership
            return wrapper;
        }

        auto vertices_ptr = reinterpret_cast<const tinybvh::bvhvec4*>(vertices_np.data());
        auto indices_ptr = static_cast<const uint32_t*>(indices_np.data());
        const auto prim_count = static_cast<uint32_t>(indices_np.shape(0));

        bvh->context = default_ctx();
        {
            nb::gil_scoped_release release;
            switch (quality) {
                case BuildQuality::Quick:
                    // tinybvh doesn't have an indexed BuildQuick so fall back to Balanced
                    bvh->Build(vertices_ptr, indices_ptr, prim_count);
                    break;
                case BuildQuality::High:
                    bvh->BuildHQ(vertices_ptr, indices_ptr, prim_count);
                    break;
                case BuildQuality::Balanced:
                default:
                    bvh->Build(vertices_ptr, indices_ptr, prim_count);
                    break;
            }
        }
        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_aabbs(
        nb::ndarray<const float, nb::c_contig> aabbs_np,
        const BuildQuality quality,
        const float traversal_cost,
        const float intersection_cost,
        const int hq_bins = HQBVHBINS) {

        if (aabbs_np.ndim() != 3 || aabbs_np.shape(1) != 2 || aabbs_np.shape(2) != 3) {
            throw std::runtime_error("Input `aabbs` must be a 3D numpy array with shape (N, 2, 3).");
        }
        if (aabbs_np.dtype() != nb::dtype<float>()) {
            throw nb::type_error("Input `aabbs` must be a float32 numpy array.");
        }

        if (quality == BuildQuality::High) {
            warnings().attr("warn")(
                "BuildQuality.High (SBVH) is not supported for AABB-based BVHs. "
                "Falling back to the standard 'Balanced' quality build.",
                userwarning()
            );
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->source_geometry_refs.append(aabbs_np);
        wrapper->quality = quality;
        wrapper->custom_type = CustomType::AABB;

        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;

        // custom intersection functions
        bvh->customIntersect = aabb_intersect_callback;
        bvh->customIsOccluded = aabb_isoccluded_callback;

        if (aabbs_np.shape(0) == 0) {
            finalize_build(wrapper, std::move(bvh));  // transfer ownership
            return wrapper;
        }

        const float* aabbs_ptr = aabbs_np.data();
        if (!aabbs_ptr) throw std::runtime_error("Internal error: AABB pointer is null.");

        const auto prim_count = static_cast<uint32_t>(aabbs_np.shape(0));

        // Pre-compute reciprocal extents for faster intersection tests
        size_t rows = prim_count, cols = 3;
        auto* inv_extents_buf = new float[rows * cols];
        nb::capsule inv_owner(inv_extents_buf, [](void* p) noexcept { delete[] static_cast<float *>(p); });
        nb::ndarray<float, nb::numpy, nb::c_contig> inv_extents_np(
            inv_extents_buf, { rows, cols }, inv_owner);

        float* inv_extents_ptr = inv_extents_buf;

        for(size_t i = 0; i < prim_count; ++i) {
            const size_t a_off = i * 6;
            const size_t i_off = i * 3;
            const float ex = aabbs_ptr[a_off + 3] - aabbs_ptr[a_off + 0];
            const float ey = aabbs_ptr[a_off + 4] - aabbs_ptr[a_off + 1];
            const float ez = aabbs_ptr[a_off + 5] - aabbs_ptr[a_off + 2];
            inv_extents_ptr[i_off + 0] = ex > 1e-6f ? 1.0f / ex : 0.0f;
            inv_extents_ptr[i_off + 1] = ey > 1e-6f ? 1.0f / ey : 0.0f;
            inv_extents_ptr[i_off + 2] = ez > 1e-6f ? 1.0f / ez : 0.0f;
        }
        // reference to reciprocals array to keep it alive
        wrapper->source_geometry_refs.append(inv_extents_np);

        AABBptrGuard guard(aabbs_ptr);

        bvh->context = default_ctx();
        {
            nb::gil_scoped_release release;
            bvh->Build(aabb_build_callback, prim_count);
        }
        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;

        // `AABBptrGuard` goes out of scope, g_build_aabbs_ptr is set back to nullptr
    }

    // Convenience builders

    static std::unique_ptr<PyBVH> from_triangles(
        const nb::ndarray<const float>& tris_np,
        const BuildQuality quality,
        const float traversal_cost,
        const float intersection_cost,
        const int hq_bins = HQBVHBINS) {

        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input `triangles` must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }
        if (tris_np.dtype() != nb::dtype<float>()) {
            throw std::runtime_error("Input `triangles` must be a float32 numpy array.");
        }

        const size_t num_tris = tris_np.shape(0);

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->quality = quality;

        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;
        bvh->hqbvhbins = hq_bins;

        if (num_tris == 0) {
            finalize_build(wrapper, std::move(bvh));  // transfer ownership
            return wrapper;
        }

        size_t rows = num_tris * 3, cols = 4;
        auto *buf = new float[rows * cols];                   // allocate
        nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<float *>(p); });

        // make a numpy-owned array
        nb::ndarray<float, nb::numpy, nb::c_contig> vertices_np(
            /*data =*/ buf,
            /*shape=*/ { rows, cols },
            /*owner=*/ owner
        );

        // reference to it in the wrapper so it doesn't get garbage collected
        wrapper->source_geometry_refs.append(vertices_np);

        // Reformat the data
        const float* tris_ptr = tris_np.data();
        float* v_ptr = vertices_np.data();
        for (size_t i = 0; i < num_tris; ++i) {
            for (size_t v = 0; v < 3; ++v) {
                const size_t v_idx = (i * 3 + v) * 4;
                const size_t t_idx = i * 9 + v * 3;
                v_ptr[v_idx + 0] = tris_ptr[t_idx + 0];
                v_ptr[v_idx + 1] = tris_ptr[t_idx + 1];
                v_ptr[v_idx + 2] = tris_ptr[t_idx + 2];
                v_ptr[v_idx + 3] = 0.0f; // Dummy w
            }
        }

        auto vertices_ptr = reinterpret_cast<const tinybvh::bvhvec4*>(vertices_np.data());

        bvh->context = default_ctx();
        {
            nb::gil_scoped_release release;
            switch (quality) {
                case BuildQuality::Quick:
                    bvh->BuildQuick(vertices_ptr, static_cast<uint32_t>(num_tris));
                    break;
                case BuildQuality::High:
                    bvh->BuildHQ(vertices_ptr, static_cast<uint32_t>(num_tris));
                    break;
                case BuildQuality::Balanced:
                default:
                    bvh->Build(vertices_ptr, static_cast<uint32_t>(num_tris));
                    break;
            }
        }
        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_points(
        const nb::ndarray<const float, nb::c_contig>& points_np,
        const float radius,
        BuildQuality quality,
        const float traversal_cost,
        const float intersection_cost,
        const int hq_bins = HQBVHBINS) {

        if (points_np.ndim() != 2 || points_np.shape(1) != 3)
            throw std::runtime_error("Input `points` must be a 2D numpy array with shape (N, 3).");
        if (points_np.dtype() != nb::dtype<float>()) {
            throw std::runtime_error("Input `points` must be a float32 numpy array.");
        }
        if (radius <= 0.0f) {
            throw std::runtime_error("Point radius must be positive.");
        }

        if (quality == BuildQuality::High) {
            warnings().attr("warn")(
                "BuildQuality.High (SBVH) is not supported for points-based BVHs. "
                "Falling back to the standard 'Balanced' quality build.",
                userwarning()
            );
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->quality = quality;
        wrapper->source_geometry_refs.append(points_np); // Store original points
        wrapper->custom_type = CustomType::Sphere;
        wrapper->sphere_radius = radius;

        bvh->customIntersect = sphere_intersect_callback;
        bvh->customIsOccluded = sphere_isoccluded_callback;
        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;

        if (points_np.shape(0) == 0) {
            finalize_build(wrapper, std::move(bvh));  // transfer ownership
            return wrapper;
        }

        const size_t num_points = points_np.shape(0);
        const float* points_ptr = points_np.data();

        // Still need to create AABBs for the build process
        size_t n = num_points;
        auto* aabbs_buf = new float[n * 2 * 3];
        nb::capsule aabbs_owner(aabbs_buf, [](void* p) noexcept { delete[] static_cast<float *>(p); });
        nb::ndarray<float, nb::numpy, nb::c_contig> aabbs_np(
            /* data */aabbs_buf,
            /* shape */ { n, static_cast<size_t>(2), static_cast<size_t>(3) },
            /* owner */ aabbs_owner);

        float* aabbs_ptr = aabbs_buf;
        for (size_t i = 0; i < num_points; ++i) {
             const size_t p_off = i * 3, a_off = i * 6;
             aabbs_ptr[a_off + 0] = points_ptr[p_off + 0] - radius;
             aabbs_ptr[a_off + 1] = points_ptr[p_off + 1] - radius;
             aabbs_ptr[a_off + 2] = points_ptr[p_off + 2] - radius;
             aabbs_ptr[a_off + 3] = points_ptr[p_off + 0] + radius;
             aabbs_ptr[a_off + 4] = points_ptr[p_off + 1] + radius;
             aabbs_ptr[a_off + 5] = points_ptr[p_off + 2] + radius;
        }

        // Even though this is for spheres, the build process uses AABBs
        AABBptrGuard guard(aabbs_ptr);

        bvh->context = default_ctx();
        {
            nb::gil_scoped_release release;
            bvh->Build(aabb_build_callback, static_cast<uint32_t>(num_points));
        }
       finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;
    }

    // Interseciton methods

    float intersect(PyRay &py_ray) const {
        // Guards
        if (!supports_layout(active_layout_, /*for_traversal=*/true)) {
            throw std::runtime_error(
                std::string("Intersection is not supported for ") +
                layout_to_string(active_layout_) + " layout: " +
                explain_requirement(active_layout_));
        }
        if (custom_type != CustomType::None && active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Custom geometry (AABBs / Spheres) can only be traversed in the Standard layout.");
        }
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            return INFINITY;
        }

        tinybvh::Ray ray(
            py_ray.origin,
            py_ray.direction,
            py_ray.t,
            py_ray.mask);

        // Create and populate context locally for this single call
        CustomGeometryContext context;
        if (custom_type == CustomType::AABB) {
            context.aabbs_ptr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]).data();
            context.inv_extents_ptr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[1]).data();
            ray.hit.auxData = &context;
        } else if (custom_type == CustomType::Sphere) {
            context.points_ptr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]).data();
            context.sphere_radius = sphere_radius;
            ray.hit.auxData = &context;
        }

        // Determine TLAS vs BLAS (only standard BVH exposes isTLAS())
        bool is_tlas = false;
        dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
            bvh.Intersect(ray);
            if constexpr (std::is_same_v<std::decay_t<decltype(bvh)>, tinybvh::BVH>) {
                is_tlas = bvh.isTLAS();
            }
        });

        {
            nb::gil_scoped_release release;
            if (ray.hit.t < py_ray.t) {
                py_ray.t = ray.hit.t; py_ray.u = ray.hit.u; py_ray.v = ray.hit.v;
#if INST_IDX_BITS == 32
                py_ray.prim_id = ray.hit.prim;
                py_ray.inst_id = is_tlas ? ray.hit.inst : static_cast<uint32_t>(-1);
#else
                const uint32_t inst = ray.hit.prim >> INST_IDX_SHFT;
                py_ray.prim_id = ray.hit.prim & PRIM_IDX_MASK;
                py_ray.inst_id = is_tlas ? inst : static_cast<uint32_t>(-1);
#endif
                return ray.hit.t;
            }
            return INFINITY;
        }
    }

    [[nodiscard]] nb::object intersect_batch(
        const nb::ndarray<const float>& origins_np,
        const nb::ndarray<const float>& directions_np,
        const nb::object& t_max_obj, const nb::object& masks_obj,
        PacketMode packet = PacketMode::Auto,
        float same_origin_eps = 1e-6f,
        float max_cone_deg = 1.0f,
        // float tmax_band_ratio = 8.0f,
        bool warn_on_incoherent = true) const
    {
        // Guards
        if (!supports_layout(active_layout_, /*for_traversal=*/true)) {
            throw std::runtime_error(
                std::string("Intersection is not supported for ") +
                layout_to_string(active_layout_) + " layout: " +
                explain_requirement(active_layout_));
        }
        if (custom_type != CustomType::None && active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Custom geometry (AABBs / Spheres) can only be traversed in the Standard layout.");
        }

        // Shape checks
        if (origins_np.ndim() != 2 || origins_np.shape(1) != 3)
            throw std::runtime_error("Origins must be a 2D numpy array with shape (N, 3).");
        if (directions_np.ndim() != 2 || directions_np.shape(1) != 3)
            throw std::runtime_error("Directions must be a 2D numpy array with shape (N, 3).");

        if (origins_np.shape(0) != directions_np.shape(0))
            throw std::runtime_error("Origins and directions must have the same length (N).");

        const Py_ssize_t n_rays = origins_np.shape(0);

        // Strides (in floats) for row/col-major handling
        const auto o_row = static_cast<size_t>(origins_np.stride(0));
        const auto o_col = static_cast<size_t>(origins_np.stride(1));
        const auto d_row = static_cast<size_t>(directions_np.stride(0));
        const auto d_col = static_cast<size_t>(directions_np.stride(1));

        // Early-out: empty batch
        if (n_rays == 0) {
            nb::object arr = np().attr("empty")(nb::make_tuple(0), "dtype"_a = HITREC_DTYPE());
            return arr;
        }

        // Optional inputs
        const float* t_max_ptr = nullptr;
        const uint32_t* masks_ptr = nullptr;

        if (!t_max_obj.is_none()) {
            auto t_max_np = nb::cast<nb::ndarray<const float>>(t_max_obj);

            if (t_max_np.ndim() != 1 || t_max_np.size() != static_cast<Py_ssize_t>(n_rays))
                throw std::runtime_error("t_max must be a 1D float32 array of length N.");

            t_max_ptr = t_max_np.data();
        }

        if (!masks_obj.is_none()) {
            auto masks_np = nb::cast<nb::ndarray<const uint32_t>>(masks_obj);

            if (masks_np.ndim() != 1 || masks_np.size() != static_cast<Py_ssize_t>(n_rays))
                throw std::runtime_error("masks must be a 1D uint32 array of length N.");

            masks_ptr = masks_np.data();
        }

        // 64B-aligned ray storage for packet kernels
        std::unique_ptr<tinybvh::Ray, void(*)(tinybvh::Ray*)> rays(
            static_cast<tinybvh::Ray*>(tinybvh::malloc64(static_cast<size_t>(n_rays) * sizeof(tinybvh::Ray), nullptr)),
            [](tinybvh::Ray* p){ if (p) tinybvh::free64(p); }
        );
        if (!rays) throw std::bad_alloc();

        // Custom geometry context: keep read-only pointers for traversal
        CustomGeometryContext context{};
        if (custom_type == CustomType::AABB) {
            auto aabbs_arr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]);
            auto inv_extents_arr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[1]);

            context.aabbs_ptr = aabbs_arr.data();
            context.inv_extents_ptr = inv_extents_arr.data();

        } else if (custom_type == CustomType::Sphere) {
            auto points_arr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]);

            context.points_ptr = points_arr.data();
            context.sphere_radius = sphere_radius;
        }

        // Early out: empty BVH
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            nb::object arr  = np().attr("empty")(nb::make_tuple(n_rays), "dtype"_a = HITREC_DTYPE());
            nb::object view = arr.attr("view")(np().attr("uint8"));
            auto bytes = nb::cast<nb::ndarray<uint8_t, nb::c_contig>>(view);
            auto* dst = reinterpret_cast<HitRecord*>(bytes.data());
            for (Py_ssize_t i = 0; i < n_rays; ++i) {
                dst[i] = { static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), INFINITY, 0.0f, 0.0f };
            }
            return arr;
        }

        const auto* origins_ptr = origins_np.data();
        const auto* directions_ptr = directions_np.data();

        // Decide base eligibility for packets (global capability / layout)
        const bool layout_is_std_bvh = (active_bvh_->layout == tinybvh::BVHBase::LAYOUT_BVH);
        bool want_packets = (packet != PacketMode::Never)
                         && layout_is_std_bvh
                         && (custom_type == CustomType::None)
                         && !is_tlas()
                         && (n_rays >= 256);

        // Warn if the whole batch mixes origins. We still decide per 256-chunk below
        if (want_packets && !same_origin_ok(origins_ptr, o_row, o_col, n_rays, same_origin_eps)) {
            if (warn_on_incoherent && packet != PacketMode::Never) {
                warnings().attr("warn")(
                    packet == PacketMode::Force ?
                    "pytinybvh: Forcing packet traversal, but ray origins differ. "
                    "Packet kernels assume a shared origin per 16x16 tile. Only rays that satisfy it will be packet-chunked."
                    :
                    "pytinybvh: Ray origins differ. Packet traversal will only apply to 16x16 chunks with shared origin."
                );
            }
        }

        // Per-block gate: rays cone size + tmax band (soft heuristic used only in Auto)
        const float tmax_band_ratio = 8.0f; // currently unused
        const float cos_thresh = std::cos(max_cone_deg * static_cast<float>(PI) / 180.0f);
        PacketGate gate{ directions_ptr, d_row, d_col, t_max_ptr, cos_thresh, tmax_band_ratio };

        {
            nb::gil_scoped_release release;

            // Construct rays
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (Py_ssize_t i = 0; i < n_rays; ++i) {
                const size_t iSo = static_cast<size_t>(i) * o_row;
                const size_t iSd = static_cast<size_t>(i) * d_row;
                const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                const uint32_t mask = masks_ptr ? masks_ptr[i] : 0xFFFFu;

                tinybvh::Ray& r = rays.get()[i];
                r = tinybvh::Ray(
                    tinybvh::bvhvec3(
                        origins_ptr[iSo + 0 * o_col],
                        origins_ptr[iSo + 1 * o_col],
                        origins_ptr[iSo + 2 * o_col]),
                    tinybvh::bvhvec3(
                        directions_ptr[iSd + 0 * d_col],
                        directions_ptr[iSd + 1 * d_col],
                        directions_ptr[iSd + 2 * d_col]),
                    /* t */ t_init,
                    /* mask */ mask
                );

                // if using custom geometry we also need the read-only context pointer
                if (custom_type != CustomType::None) {
                    r.hit.auxData = &context;
                }
            }

            dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
                using BVHType = std::decay_t<decltype(bvh)>;

                if constexpr (std::is_same_v<BVHType, tinybvh::BVH>) {
                    // Standard layout: we may use packet traversal
                    if (want_packets) {
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (Py_ssize_t i = 0; i < n_rays; i += 256) {
                            const Py_ssize_t end = std::min(i + 256, n_rays);
                            bool used_packet = false;

                            if (end - i == 256) {
                                const bool chunk_same_origin = same_origin_ok(
                                    origins_ptr + static_cast<size_t>(i) * o_row,
                                    o_row, o_col, 256, same_origin_eps);
                                const bool chunk_coherent = (packet == PacketMode::Force)
                                    ? true
                                    : (gate.cone_ok(i, 256) && PacketGate::tmax_ok(i, 256));

                                if (chunk_same_origin && chunk_coherent) {
                                #if defined(BVH_USEAVX) && defined(BVH_USESSE)
                                    bvh.Intersect256RaysSSE(&rays.get()[i]);
                                #else
                                    bvh.Intersect256Rays(&rays.get()[i]);
                                #endif
                                    used_packet = true;
                                }
                            }
                            if (!used_packet) {
                                // Fallback: scalar traversal for this chunk
                                for (Py_ssize_t j = i; j < end; ++j) {
                                    bvh.Intersect(rays.get()[j]);
                                }
                            }
                        }
                    } else {
                        // Non-packet (scalar) traversal
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (Py_ssize_t i = 0; i < n_rays; ++i) bvh.Intersect(rays.get()[i]);
                    }
                } else {
                    // Non-standard layouts: scalar traversal only
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(dynamic)
                    #endif
                    for (Py_ssize_t i = 0; i < n_rays; ++i) bvh.Intersect(rays.get()[i]);
                }
            });
        } // GIL re-acquired

        // Determine TLAS vs BLAS (only standard BVH exposes isTLAS())
        bool is_tlas = false;
        dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
            if constexpr (std::is_same_v<std::decay_t<decltype(bvh)>, tinybvh::BVH>)
                is_tlas = bvh.isTLAS();
        });

        // Allocate final result (structured dtype)
        nb::object arr = np().attr("empty")(nb::make_tuple(n_rays), "dtype"_a = HITREC_DTYPE());

        // Get byte view so we can treat it as HitRecord*
        nb::object view = arr.attr("view")(np().attr("uint8"));
        auto bytes = nb::cast<nb::ndarray<uint8_t, nb::c_contig>>(view);
        auto* dst = reinterpret_cast<HitRecord*>(bytes.data());

        // Convert rays to HitRecord directly into the output
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (Py_ssize_t i = 0; i < n_rays; ++i) {
            const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;

            if (rays.get()[i].hit.t < t_init) {
                #if INST_IDX_BITS == 32
                const uint32_t prim = rays.get()[i].hit.prim;
                const uint32_t inst = is_tlas ? rays.get()[i].hit.inst : static_cast<uint32_t>(-1);
                #else
                const uint32_t prim = rays.get()[i].hit.prim & PRIM_IDX_MASK;
                const uint32_t inst = is_tlas ? (rays.get()[i].hit.prim >> INST_IDX_SHFT) : static_cast<uint32_t>(-1);
                #endif
                dst[i] = { prim, inst, rays.get()[i].hit.t, rays.get()[i].hit.u, rays.get()[i].hit.v };
            } else {
                dst[i] = { static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), INFINITY, 0.0f, 0.0f };
            }
        }

        return arr;
    }

    [[nodiscard]] bool is_occluded(const PyRay &py_ray) const {

        // Guards
        if (!supports_layout(active_layout_, /*for_traversal=*/true)) {
            throw std::runtime_error(
                std::string("Intersection is not supported for ") +
                layout_to_string(active_layout_) + " layout: " +
                explain_requirement(active_layout_));
        }
        if (custom_type != CustomType::None && active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Custom geometry (AABBs / Spheres) can only be traversed in the Standard layout.");
        }
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            return false; // nothing to occlude
        }

        tinybvh::Ray ray(
            py_ray.origin,
            py_ray.direction,
            py_ray.t,   // the ray's t is used as the maximum distance for the occlusion check
            py_ray.mask);

        // Create and populate context locally for this single call
        CustomGeometryContext context;
        if (custom_type == CustomType::AABB) {
            context.aabbs_ptr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]).data();
            context.inv_extents_ptr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[1]).data();
            ray.hit.auxData = &context;
        } else if (custom_type == CustomType::Sphere) {
            context.points_ptr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]).data();
            context.sphere_radius = sphere_radius;
            ray.hit.auxData = &context;
        }

        {
            nb::gil_scoped_release release;

            return dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
                return bvh.IsOccluded(ray);
            });
        }
    }

    [[nodiscard]] nb::ndarray<bool> is_occluded_batch(
        const nb::ndarray<const float>& origins_np,
        const nb::ndarray<const float>& directions_np,
        const nb::object& t_max_obj, const nb::object& masks_obj,
        PacketMode packet = PacketMode::Auto,
        float same_origin_eps = 1e-6f,
        float max_cone_deg = 1.0f,
        // float tmax_band_ratio = 8.0f,
        bool warn_on_incoherent = true) const
    {
        // Guards
        if (!supports_layout(active_layout_, /*for_traversal=*/true)) {
            throw std::runtime_error(
                std::string("Intersection is not supported for ") +
                layout_to_string(active_layout_) + " layout: " +
                explain_requirement(active_layout_));
        }
        if (custom_type != CustomType::None && active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Custom geometry (AABBs / Spheres) can only be traversed in the Standard layout.");
        }

        // Shape checks
        if (origins_np.ndim() != 2 || origins_np.shape(1) != 3)
            throw std::runtime_error("Origins must be a 2D numpy array with shape (N, 3).");
        if (directions_np.ndim() != 2 || directions_np.shape(1) != 3)
            throw std::runtime_error("Directions must be a 2D numpy array with shape (N, 3).");

        if (origins_np.shape(0) != directions_np.shape(0))
            throw std::runtime_error("Origins and directions must have the same length (N).");

        const Py_ssize_t n_rays = origins_np.shape(0);

        // Strides (in floats) for row/col-major handling
        const auto o_row = static_cast<size_t>(origins_np.stride(0));
        const auto o_col = static_cast<size_t>(origins_np.stride(1));
        const auto d_row = static_cast<size_t>(directions_np.stride(0));
        const auto d_col = static_cast<size_t>(directions_np.stride(1));

        // Prepare output array
        nb::object result_obj = np().attr("empty")(nb::make_tuple(n_rays), "dtype"_a = np().attr("bool_"));
        auto result = nb::cast<nb::ndarray<bool>>(result_obj);

        // Ensure initial state (False for all)
        auto* outp = result.data();
        std::fill_n(outp, n_rays, false);

        // Early-out: empty batch
        if (n_rays == 0) return result;

        // Optional inputs
        const float* t_max_ptr  = nullptr;
        const uint32_t* masks_ptr  = nullptr;

        if (!t_max_obj.is_none()) {
            auto t_max_np = nb::cast<nb::ndarray<const float>>(t_max_obj);
            if (t_max_np.ndim() != 1 || t_max_np.size() != n_rays)
                throw std::runtime_error("t_max must be a 1D float32 array of length N.");
            t_max_ptr = t_max_np.data();
        }
        if (!masks_obj.is_none()) {
            auto masks_np = nb::cast<nb::ndarray<const uint32_t>>(masks_obj);
            if (masks_np.ndim() != 1 || masks_np.size() != n_rays)
                throw std::runtime_error("masks must be a 1D uint32 array of length N.");
            masks_ptr = masks_np.data();
        }

        // Custom geometry context: keep read-only pointers for traversal
        CustomGeometryContext context{};
        if (custom_type == CustomType::AABB) {
            auto aabbs_arr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]);
            auto inv_extents_arr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[1]);

            context.aabbs_ptr = aabbs_arr.data();
            context.inv_extents_ptr = inv_extents_arr.data();

        } else if (custom_type == CustomType::Sphere) {
            auto points_arr = nb::cast<nb::ndarray<const float>>(source_geometry_refs[0]);

            context.points_ptr = points_arr.data();
            context.sphere_radius = sphere_radius;
        }

        // Early out: empty BVH
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            return result;
        }

        const auto* origins_ptr = origins_np.data();
        const auto* directions_ptr = directions_np.data();

        // Decide base eligibility for packets (global capability / layout)
        const bool layout_is_std_bvh = (active_bvh_->layout == tinybvh::BVHBase::LAYOUT_BVH);
        bool want_packets = (packet != PacketMode::Never)
                         && layout_is_std_bvh
                         && (custom_type == CustomType::None)
                         && !is_tlas()
                         && (n_rays >= 256);

        // Warn if the whole batch mixes origins. We still decide per 256-chunk below
        if (want_packets && !same_origin_ok(origins_ptr, o_row, o_col, n_rays, same_origin_eps)) {
            if (warn_on_incoherent && packet != PacketMode::Never) {
                warnings().attr("warn")(
                    packet == PacketMode::Force ?
                    "pytinybvh: Forcing packet traversal, but ray origins differ. "
                    "Packet kernels assume a shared origin per 16x16 tile. Only rays that satisfy it will be packet-chunked."
                    :
                    "pytinybvh: Ray origins differ. Packet traversal will only apply to 16x16 chunks with shared origin."
                );
            }
        }

        // Per-block gate: rays cone size + tmax band (soft heuristic used only in Auto)
        const float tmax_band_ratio = 8.0f; // currently unused
        const float cos_thresh = std::cos(max_cone_deg * static_cast<float>(PI) / 180.0f);
        PacketGate gate{ directions_ptr, d_row, d_col, t_max_ptr, cos_thresh, tmax_band_ratio };

        // 64B-aligned ray storage for packet kernels
        std::unique_ptr<tinybvh::Ray, void(*)(tinybvh::Ray*)> rays(
            static_cast<tinybvh::Ray*>(tinybvh::malloc64(static_cast<size_t>(n_rays) * sizeof(tinybvh::Ray), nullptr)),
            [](tinybvh::Ray* p){ if (p) tinybvh::free64(p); }
        );
        if (!rays) throw std::bad_alloc();

        // Scratch for occlusion result (byte per ray, converted to bools at the end)
        std::vector<uint8_t> occluded(static_cast<size_t>(n_rays), 0);

        {
            nb::gil_scoped_release release;

            // Construct rays
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (Py_ssize_t i = 0; i < n_rays; ++i) {
                const size_t iSo = static_cast<size_t>(i) * o_row;
                const size_t iSd = static_cast<size_t>(i) * d_row;
                const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                const uint32_t mask = masks_ptr ? masks_ptr[i] : 0xFFFFu;

                tinybvh::Ray& r = rays.get()[i];
                r = tinybvh::Ray(
                    /* origin */ tinybvh::bvhvec3(
                        origins_ptr[iSo + 0 * o_col],
                        origins_ptr[iSo + 1 * o_col],
                        origins_ptr[iSo + 2 * o_col]
                        ),
                    /* direction */ tinybvh::bvhvec3(
                        directions_ptr[iSd + 0 * d_col],
                        directions_ptr[iSd + 1 * d_col],
                        directions_ptr[iSd + 2 * d_col]
                        ),
                    /* t */ t_init,
                    /* mask */ mask
                );

                // if using custom geometry we also need the read-only context pointer
                if (custom_type != CustomType::None) {
                    r.hit.auxData = &context;
                }
            }

            dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
                using BVHType = std::decay_t<decltype(bvh)>;

                if constexpr (std::is_same_v<BVHType, tinybvh::BVH>) { // BLAS only
                    if (want_packets) {
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (Py_ssize_t i = 0; i < n_rays; i += 256) {
                            const Py_ssize_t end = std::min(i + 256, n_rays);
                            bool used_packet = false;

                            if (end - i == 256) {
                                const bool chunk_same_origin = same_origin_ok(
                                    origins_ptr + static_cast<size_t>(i) * o_row,
                                    o_row, o_col, 256, same_origin_eps);
                                const bool chunk_coherent = (packet == PacketMode::Force)
                                    ? true
                                    : (gate.cone_ok(i, 256) && PacketGate::tmax_ok(i, 256));

                                if (chunk_same_origin && chunk_coherent) {
                                #if defined(BVH_USEAVX) && defined(BVH_USESSE)
                                    bvh.Intersect256RaysSSE(&rays.get()[i]);
                                #else
                                    bvh.Intersect256Rays(&rays.get()[i]);
                                #endif
                                    for (Py_ssize_t j = i; j < end; ++j) {
                                        const float t_init = t_max_ptr ? t_max_ptr[j] : 1e30f;
                                        if (rays.get()[j].hit.t < t_init) occluded[static_cast<size_t>(j)] = 1;
                                    }
                                    used_packet = true;
                                }
                            }
                            if (!used_packet) {
                                // Fallback: scalar occlusion for this chunk
                                for (Py_ssize_t j = i; j < end; ++j)
                                    if (bvh.IsOccluded(rays.get()[j])) occluded[static_cast<size_t>(j)] = 1;
                            }
                        }
                    } else {
                        // Scalar occlusion
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (Py_ssize_t j = 0; j < n_rays; ++j)
                            if (bvh.IsOccluded(rays.get()[j])) occluded[static_cast<size_t>(j)] = 1;
                    }
                } else {
                    // Non-standard layouts (scalar path only)
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(dynamic)
                    #endif
                    for (Py_ssize_t j = 0; j < n_rays; ++j)
                        if (bvh.IsOccluded(rays.get()[j])) occluded[static_cast<size_t>(j)] = 1;
                }
            });
        } // GIL re-acquired

        // Pack Python bools
        auto* outp_final = result.data();
        for (Py_ssize_t i = 0; i < n_rays; ++i)
            outp_final[i] = (occluded[static_cast<size_t>(i)] != 0);

        return result;
    }

    [[nodiscard]] bool intersect_sphere(const tinybvh::bvhvec3& center, float radius) const {
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            return false;
        }
        if (active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Sphere intersection is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto* bvh = static_cast<const tinybvh::BVH*>(active_bvh_);
        return bvh->IntersectSphere(center, radius);
    }

    // Refitting, TLAS

    void refit() const {
        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Refit is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);

        if (!bvh->refittable) {
            throw std::runtime_error("BVH is not refittable. This is expected for a BVH built with spatial splits (High Quality preset).");
        }
        bvh->Refit();
    }

    static std::unique_ptr<PyBVH> build_tlas(const nb::object& instances_obj, const nb::list& blases_py) {

        // Validate dtype is structured and has the fields we need
        nb::object dtype_obj  = instances_obj.attr("dtype");
        nb::object fields_obj = dtype_obj.attr("fields");  // None for non-structured
        if (fields_obj.is_none())
            throw std::runtime_error("Input `instances` must be a 1D structured numpy array.");

        nb::dict fields;
        if (nb::isinstance<nb::dict>(fields_obj)) {
            fields = nb::cast<nb::dict>(fields_obj);
        } else {
            // Make a real dict from any mapping-like object
            fields = nb::cast<nb::dict>(builtins().attr("dict")(fields_obj));
        }

        if (!fields.contains("transform") || !fields.contains("blas_id"))
            throw std::runtime_error("Instance dtype must contain 'transform' and 'blas_id'.");

        if (nb::cast<int>(instances_obj.attr("ndim")) != 1)
            throw std::runtime_error("Instances must be a 1D array.");

        auto inst_count = nb::cast<Py_ssize_t>(instances_obj.attr("shape")[0]);

        if (inst_count == 0)
            return std::make_unique<PyBVH>(); // empty TLAS

        // uint8 view to get a contiguous byte pointer
        nb::object view_u8 = instances_obj.attr("view")(np().attr("uint8"));
        auto bytes = nb::cast<nb::ndarray<uint8_t, nb::c_contig>>(view_u8);

        // Get info for manual structured array traversal
        const char* base_ptr = reinterpret_cast<const char*>(bytes.data());
        const auto itemsize = nb::cast<size_t>(dtype_obj.attr("itemsize"));

        // Get field offsets from the dtype
        auto get_offset = [&](const char* name) {
            auto info = nb::cast<nb::tuple>(fields[name]);
            return nb::cast<size_t>(info[1]);
        };

        const size_t transform_offset = get_offset("transform");
        const size_t blas_id_offset = get_offset("blas_id");
        const size_t mask_offset = fields.contains("mask") ? get_offset("mask") : static_cast<size_t>(-1);

        // Create wrapper that will hold the TLAS
        auto wrapper = std::make_unique<PyBVH>();

        // Prepare data holders
        wrapper->instances_data.resize(inst_count);
        wrapper->blas_pointers.reserve(blases_py.size());

        // Extract the BVHBase* pointers from the BLASes
        for (const auto& blas_obj : blases_py) {
            auto& py_blas = nb::cast<PyBVH&>(blas_obj);

            if (!py_blas.active_bvh_) {
                throw std::runtime_error("One of the provided BLASes is uninitialized.");
            }
            // this gets the BVHBase* raw pointer regardless of the BLAS's actual layout
            wrapper->blas_pointers.push_back(py_blas.active_bvh_);
        }

        // Populate instance data by directly accessing memory
        for (Py_ssize_t i = 0; i < inst_count; ++i) {
            const char* record_ptr = base_ptr + i * itemsize;

            const auto* transform_ptr = reinterpret_cast<const float*>(record_ptr + transform_offset);
            const uint32_t blas_id = *reinterpret_cast<const uint32_t*>(record_ptr + blas_id_offset);

            wrapper->instances_data[i].blasIdx = blas_id;
            if (mask_offset != static_cast<size_t>(-1)) {
                wrapper->instances_data[i].mask = *reinterpret_cast<const uint32_t*>(record_ptr + mask_offset);
            }
            std::memcpy(wrapper->instances_data[i].transform.cell, transform_ptr, 16 * sizeof(float));
        }

        // Keep Python objects alive to prevent their data from being garbage collected
        wrapper->source_geometry_refs.append(instances_obj);
        wrapper->source_geometry_refs.append(blases_py);

        // Build TLAS and assign it to the wrapper
        // (a TLAS is always a standard-layout tinybvh::BVH)
        auto tlas_bvh = std::make_unique<tinybvh::BVH>();

        tlas_bvh->context = default_ctx();

        {
            nb::gil_scoped_release release;
            tlas_bvh->Build(
                wrapper->instances_data.data(),
                static_cast<uint32_t>(inst_count),
                wrapper->blas_pointers.data(),
                static_cast<uint32_t>(wrapper->blas_pointers.size())
            );
        }

        finalize_build(wrapper, std::move(tlas_bvh)); // transfer ownership
        return wrapper;
    }

    // Load / save

    static std::unique_ptr<PyBVH> load(
        const nb::object& filepath_obj,
        const nb::object& vertices_obj_in,
        const nb::object& indices_obj)
    {
        const auto filepath = nb::cast<std::string>(nb::str(filepath_obj));

        // Normalize vertices to float32, C-contig (V, 4)
        nb::object vertices_obj = np().attr("ascontiguousarray")(vertices_obj_in, "dtype"_a = np().attr("float32"));
        auto verts = nb::cast<nb::ndarray<>>(vertices_obj);
        if (verts.ndim() != 2 || verts.shape(1) != 4)
            throw std::runtime_error("Input `vertices` must have shape (V, 4) and dtype float32.");

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh     = std::make_unique<tinybvh::BVH>();
        bool success = false;

        // Keep vertices alive
        wrapper->source_geometry_refs.append(vertices_obj);

        // Pointer to vertex data
        const auto* vptr = reinterpret_cast<const tinybvh::bvhvec4*>(verts.data());
        const auto V = static_cast<uint32_t>(verts.shape(0));

        if (indices_obj.is_none()) {
            // Non-indexed geometry

            if (V % 3 != 0)
                throw std::runtime_error("Loading for non-indexed geometry expects vertex count to be divisible by 3.");

            const uint32_t prim_count = V / 3;
            success = bvh->Load(filepath.c_str(), vptr, prim_count);
        } else {
            // Indexed geometry: normalize to uint32, C-contig (N, 3)

            nb::object indices_norm = np().attr("ascontiguousarray")(indices_obj, "dtype"_a = np().attr("uint32"));
            auto idx = nb::cast<nb::ndarray<>>(indices_norm);
            if (idx.ndim() != 2 || idx.shape(1) != 3)
                throw std::runtime_error("Input `indices` must have shape (N, 3) and dtype uint32.");

            // Cheap safety: bounds check
            {
                const auto* ip = static_cast<const uint32_t*>(idx.data());
                const size_t n3 = static_cast<size_t>(idx.shape(0)) * 3;
                for (size_t k = 0; k < n3; ++k)
                    if (ip[k] >= V)
                        throw std::runtime_error("Input `indices` contains an out-of-range vertex index.");
            }

            // Keep indices alive
            wrapper->source_geometry_refs.append(indices_norm);

            const auto* iptr = static_cast<const uint32_t*>(idx.data());
            const auto prim_count = static_cast<uint32_t>(idx.shape(0));
            success = bvh->Load(filepath.c_str(), vptr, iptr, prim_count);
        }

        if (!success)
            throw std::runtime_error("Failed to load BVH (file incompatible or geometry mismatch).");

        // Infer quality
        wrapper->quality = bvh->refittable ? BuildQuality::Balanced : BuildQuality::High;

        finalize_build(wrapper, std::move(bvh));
        return wrapper;
    }

    void save(const nb::object& filepath_obj) const {

        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Saving is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);

        // Accept str / Path / anything with __fspath__
        auto filepath = nb::cast<std::string>(nb::str(filepath_obj));
        bvh->Save(filepath.c_str());
    }

    // Advanced manipulation methods

    void optimize(unsigned int iterations, bool extreme, bool stochastic) {

        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Optimization is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);

        tinybvh::BVH_Verbose verbose_bvh;

        verbose_bvh.bvhNode = static_cast<tinybvh::BVH_Verbose::BVHNode*>(verbose_bvh.AlignedAlloc(
            sizeof(tinybvh::BVH_Verbose::BVHNode) * bvh->triCount * 3));

        verbose_bvh.allocatedNodes = bvh->triCount * 3;
        verbose_bvh.ConvertFrom(*bvh);

        verbose_bvh.Optimize(iterations, extreme, stochastic);

        bvh->ConvertFrom(verbose_bvh, true /* compact */);

        quality = bvh->refittable ? BuildQuality::Balanced : BuildQuality::High;
    }

    void set_opacity_maps(const nb::ndarray<uint32_t, nb::c_contig>& map_data, uint32_t N) {
        // tinybvh expects a specific size: N*N bits per triangle
        // The array should contain (triCount * N * N) / 32 uint32_t values
        const size_t expected_size = (active_bvh_->triCount * N * N + 31) / 32;
        if (map_data.size() != expected_size) {
            throw std::runtime_error("Opacity map data has incorrect size for the given N and primitive count.");   // TODO this is not a very helpful error
        }
        opacity_map_ref = map_data; // Keep reference
        active_bvh_->SetOpacityMicroMaps(opacity_map_ref.data(), N);
    }

    void compact() const {
        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Compaction is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);
        bvh->Compact();
    }

    void split_leaves(uint32_t max_prims) const {
        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Splitting leaves is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);
        if (bvh->triCount == 0) throw std::runtime_error("BVH is not initialized.");
        if (bvh->usedNodes + bvh->triCount > bvh->allocatedNodes) {
            throw std::runtime_error("Cannot split leaves: not enough allocated node capacity.");
        }
        bvh->SplitLeafs(max_prims);
    }

    void combine_leaves() const {
        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Combining leaves is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);
        if (bvh->triCount == 0) throw std::runtime_error("BVH is not initialized.");
        bvh->CombineLeafs();
    }

    [[nodiscard]] bool is_tlas() const {
        if (!active_bvh_) return false;

        return dispatch_bvh_call(active_bvh_, [](auto& bvh){
            using BvhType = std::decay_t<decltype(bvh)>;
            // Only the standard BVH layout can be a TLAS
            if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                return bvh.isTLAS();
            }

            return false; // other layouts are always BLASes
        });
    }

    static nb::object get_nodes(const PyBVH &self) {
        // TODO: Apply this zero-copy approach to the other properties where possible

        // if no BVH or not standard layout: return empty structured array
        if (!self.active_bvh_ || self.active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            nb::object empty = np()
                .attr("empty")(nb::make_tuple(0), "dtype"_a = BVHNODE_DTYPE());
            empty.attr("setflags")("write"_a = false);
            return empty;
        }

        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto* bvh = static_cast<const tinybvh::BVH*>(self.active_bvh_);
        if (!bvh) {
            nb::object empty = np()
                .attr("empty")(nb::make_tuple(0), "dtype"_a = BVHNODE_DTYPE());
            empty.attr("setflags")("write"_a = false);
            return empty;
        }

        size_t n = bvh->usedNodes;
        auto itemsize = nb::cast<size_t>(BVHNODE_DTYPE().attr("itemsize"));
        const auto* data = reinterpret_cast<const uint8_t*>(bvh->bvhNode);

        // numpy array of bytes that aliases the nodes data
        auto bytes = nb::ndarray<const uint8_t, nb::numpy, nb::c_contig>(
            data,
            /* shape  */ { n * itemsize },
            /* owner  */ nb::find(&self)
        );

        // Materialize as a Python object and reinterpret to our custom dtype
        nb::object np_bytes = bytes.cast();
        nb::object arr = np_bytes
            .attr("view")(BVHNODE_DTYPE())
            .attr("reshape")(nb::make_tuple(n));

        arr.attr("setflags")("write"_a = false);    // just to be sure

        return arr;
    }

    static nb::ndarray<const uint32_t, nb::numpy, nb::c_contig, nb::ro> get_prims_indices(const PyBVH &self) {

        // if no BVH or not standard layout: return empty uint32 array
        if (!self.active_bvh_ || self.active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH)
            return {};

        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto* bvh = static_cast<const tinybvh::BVH*>(self.active_bvh_);
        if (!bvh->primIdx || bvh->idxCount == 0)
            return {};

        auto arr = nb::ndarray<const uint32_t, nb::numpy, nb::c_contig, nb::ro>(
            bvh->primIdx,
            { static_cast<size_t>(bvh->idxCount) },
            nb::find(&self)
        );
        return arr;
    }

    [[nodiscard]] nb::dict get_buffers() const {

        if (!active_bvh_) {
            throw std::runtime_error("BVH is not initialized.");
        }

        nb::dict buffers;
        const PyBVH& self = *this;

        // Get buffers for the BVH structure itself
        // all read only, all as float32 (or uint32, for prim_indices)
        dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
            using BvhType = std::decay_t<decltype(bvh)>;

            // Standard layout
            if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                if (bvh.bvhNode) {
                    // exposes nodes as a raw float array. Each is 32 bytes (8 floats)
                    buffers["nodes"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvhNode),
                        { static_cast<size_t>(bvh.usedNodes) * 8 },
                        nb::find(&self)
                    );
                }
                if (bvh.primIdx) {
                    buffers["prim_indices"] = nb::ndarray<const uint32_t, nb::ro>(
                        bvh.primIdx,
                        { static_cast<size_t>(bvh.idxCount) },
                        nb::find(&self)
                    );
                }
            }
            // GPU layouts
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH_GPU>) { // NOLINT(*-branch-clone) Reason: being explicit is better here
                if (bvh.bvhNode) {
                    // Each node is 64 bytes (16 floats)
                    buffers["nodes"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvhNode),
                        { static_cast<size_t>(bvh.usedNodes) * 16 },
                        nb::find(&self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH4_GPU>) {
                if (bvh.bvh4Data) {
                    // usedBlocks is in units of 16 bytes (a bvhvec4). Total floats = usedBlocks * 4
                    buffers["packed_data"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvh4Data),
                        { static_cast<size_t>(bvh.usedBlocks) * 4 },
                        nb::find(&self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH8_CWBVH>) {
                if (bvh.bvh8Data) {
                    // usedBlocks is in units of 16 bytes
                    buffers["nodes"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvh8Data),
                        { static_cast<size_t>(bvh.usedBlocks) * 4 },
                        nb::find(&self)
                    );
                }
                if (bvh.bvh8Tris) {
                    // this determines the size based on compilation flags
                    #ifdef CWBVH_COMPRESSED_TRIS
                        size_t elements_per_tri = 4 * 4; // 4 vec4s
                    #else
                        size_t elements_per_tri = 3 * 4; // 3 vec4s
                    #endif
                    buffers["triangles"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvh8Tris),
                        { static_cast<size_t>(bvh.triCount) * elements_per_tri },
                        nb::find(&self)
                    );
                }
            }
            // CPU SIMD layouts
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH4_CPU>) {
                if (bvh.bvh4Data) {
                    // usedBlocks is in units of 64 bytes (CacheLine). Total floats = usedBlocks * 16
                    buffers["packed_data"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvh4Data),
                        { static_cast<size_t>(bvh.usedBlocks) * 16 },
                        nb::find(&self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH8_CPU>) {
                if (bvh.bvh8Data) {
                    // usedBlocks is in units of 64 bytes (CacheLine). Total floats = usedBlocks * 16
                    buffers["packed_data"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvh8Data),
                        { static_cast<size_t>(bvh.usedBlocks) * 16 },
                        nb::find(&self)
                    );
                }
            }
            // Intermediate layouts
            else if constexpr (std::is_same_v<BvhType, tinybvh::MBVH<4>> || std::is_same_v<BvhType, tinybvh::MBVH<8>>) {
                 if (bvh.mbvhNode) {
                    // MBVH<4> node is 48 bytes (12 floats). MBVH<8> is 64 bytes (16 floats)
                    constexpr size_t node_size_in_floats = sizeof(BvhType::MBVHNode) / sizeof(float);
                    buffers["nodes"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.mbvhNode),
                        { static_cast<size_t>(bvh.usedNodes) * node_size_in_floats },
                        nb::find(&self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH_SoA>) {
                if (bvh.bvhNode) {
                    // Each node is 64 bytes (16 floats)
                    buffers["nodes"] = nb::ndarray<const float, nb::ro>(
                        reinterpret_cast<const float*>(bvh.bvhNode),
                        { static_cast<size_t>(bvh.usedNodes) * 16 },
                        nb::find(&self)
                    );
                }
            }
        });

        // Add custom geometry data if present
        switch (self.custom_type) {
            case CustomType::AABB:
                // for AABBs source_geometry_refs contains [aabbs_np, inv_extents_np]
                if (self.source_geometry_refs.size() >= 2) {
                    buffers["aabbs"] = self.source_geometry_refs[0];
                    buffers["inv_extents"] = self.source_geometry_refs[1];
                }
                break;
            case CustomType::Sphere:
                // for Spheres source_geometry_refs contains [points_np]
                if (!self.source_geometry_refs.empty()) {
                    buffers["points"] = self.source_geometry_refs[0];
                    buffers["sphere_radius"] = self.sphere_radius;
                }
                break;
            case CustomType::None: // triangles
                if (!self.source_geometry_refs.empty()) {
                    buffers["vertices"] = self.source_geometry_refs[0];
                    // if indexed mesh, also include the indices buffer
                    if (self.source_geometry_refs.size() > 1) {
                        buffers["indices"] = self.source_geometry_refs[1];
                    }
                }
                break;
        }

        return buffers;
    }

    [[nodiscard]] nb::list get_cached_layouts() const {
        nb::list cached;
        if (soa_)      cached.append(Layout::SoA);
        if (mbvh4_)    cached.append(Layout::MBVH4);
        if (mbvh8_)    cached.append(Layout::MBVH8);
        if (bvh4_cpu_) cached.append(Layout::BVH4_CPU);
        if (bvh4_gpu_) cached.append(Layout::BVH4_GPU);
        if (cwbvh_)    cached.append(Layout::CWBVH);
        if (bvh8_cpu_) cached.append(Layout::BVH8_CPU);
        if (bvh_gpu_)  cached.append(Layout::BVH_GPU);
        return cached;
    }

    [[nodiscard]] std::string get_repr() const {
        if (!active_bvh_) return "<pytinybvh.BVH (uninitialized)>";

        std::string repr = "<pytinybvh.BVH (";
        repr += std::to_string(active_bvh_->triCount) + " primitives, ";

        const auto L = active_bvh_->layout;
        const bool show_nodes = (L == tinybvh::BVHBase::LAYOUT_BVH) ||
                                (L == tinybvh::BVHBase::LAYOUT_BVH_SOA);
        if (show_nodes) {
            repr += std::to_string(active_bvh_->usedNodes) + " nodes, ";
        }

        switch (custom_type) {
        case CustomType::AABB:   repr += "Geometry: AABBs, "; break;
        case CustomType::Sphere: repr += "Geometry: Spheres, "; break;
        default: break;
        }

        if (this->is_tlas()) {
            repr += "Type: TLAS, ";
        } else {
            std::string quality_str;
            switch(quality) {
            case BuildQuality::Quick: quality_str = "Quick"; break;
            case BuildQuality::Balanced: quality_str = "Balanced"; break;
            case BuildQuality::High: quality_str = "High"; break;
            }
            repr += "Type: BLAS, Quality: " + quality_str + ", ";
        }
        repr += std::string("Layout: ") + layout_to_string(L) + ", ";
        if (active_bvh_->layout == tinybvh::BVHBase::LAYOUT_BVH) {
            repr += std::string("Status: ") + (active_bvh_->may_have_holes ? "Not compact" : "Compact");
        } else {
            repr += "Status: n/a";
        }
        repr += ")>";
        return repr;
    }
};


// =============================================== nanobind module =====================================================

NB_MAKE_OPAQUE(TLASInstance);
NB_MAKE_OPAQUE(HitRecord);

NB_MODULE(_pytinybvh, m) {
    isa_compat_check();

    m.doc() = "Python bindings for the tinybvh library";

    nb::object hr = make_hit_record_dtype();
    nb::object nd = make_bvh_node_dtype();
    nb::object id = make_instance_dtype();

    Py_INCREF(hr.ptr()); g_hit_record_dtype = hr.ptr();
    Py_INCREF(nd.ptr()); g_bvh_node_dtype = nd.ptr();
    Py_INCREF(id.ptr()); g_instance_dtype = id.ptr();

    m.attr("hit_record_dtype") = nb::borrow(g_hit_record_dtype);
    m.attr("bvh_node_dtype") = nb::borrow(g_bvh_node_dtype);
    m.attr("instance_dtype") = nb::borrow(g_instance_dtype);

    // Build quality, Geometry type, BVH Layout, Packet Mode, Cache Policy and Layout enums exposed to Python
    nb::enum_<BuildQuality>(m, "BuildQuality", "Enum for selecting BVH build quality.")
        .value("Quick", BuildQuality::Quick, "Fastest build, lower quality queries.")
        .value("Balanced", BuildQuality::Balanced, "Balanced build time and query performance (default).")
        .value("High", BuildQuality::High, "Slowest build (uses spatial splits), highest quality queries.");

    nb::enum_<GeometryType>(m, "GeometryType", "Enum for the underlying geometry type of the BVH.")
        .value("Triangles", GeometryType::Triangles, "The BVH was built over a triangle mesh.")
        .value("AABBs", GeometryType::AABBs, "The BVH was built over custom Axis-Aligned Bounding Boxes.")
        .value("Spheres", GeometryType::Spheres, "The BVH was built over a point cloud with a radius (spheres).");

    nb::enum_<CachePolicy>(m, "CachePolicy", "Enum for managing cached BVH layouts after conversion.")
        .value("ActiveOnly", CachePolicy::ActiveOnly, "(Default) Free memory of any non-active layouts after a conversion. This minimizes memory usage.")
        .value("All", CachePolicy::All, "Keep all generated layouts in memory. This uses more memory but makes switching back to a previously used layout instantaneous.");

    nb::enum_<PacketMode>(m, "PacketMode", "Enum for controlling SIMD packet traversal in batched queries.")
        .value("Auto",  PacketMode::Auto,  "Use packets for coherent rays with a shared origin.")
        .value("Never", PacketMode::Never, "Always use scalar traversal. Safest for non-coherent rays.")
        .value("Force", PacketMode::Force, "Force packet traversal. Unsafe for non-coherent rays.");

    nb::enum_<Layout>(m, "Layout", "Enum for the internal memory layout of the BVH.")
        .value("Standard", Layout::Standard, "Standard BVH layout. Always available and traversable.")
        .value("SoA", Layout::SoA, "Structure of Arrays layout, optimized for AVX/NEON traversal.")
        .value("BVH_GPU", Layout::BVH_GPU, "Aila & Laine layout, optimized for GPU traversal (scalar traversal on CPU).")
        .value("MBVH4", Layout::MBVH4, "4-wide MBVH layout. This is a structural format used as an intermediate for other wide layouts. Not directly traversable.")
        .value("MBVH8", Layout::MBVH8, "8-wide MBVH layout. This is a structural format used as an intermediate for other wide layouts. Not directly traversable.")
        .value("BVH4_CPU", Layout::BVH4_CPU, "4-wide BVH layout, optimized for SSE CPU traversal.")
        .value("BVH4_GPU", Layout::BVH4_GPU, "Quantized 4-wide BVH layout for GPUs (scalar traversal on CPU).")
        .value("CWBVH", Layout::CWBVH, "Compressed 8-wide BVH layout, optimized for AVX traversal.")
        .value("BVH8_CPU", Layout::BVH8_CPU, "8-wide BVH layout, optimized for AVX2 CPU traversal.");

    // Top-level functions

    m.def("hardware_info", &get_hardware_info,
        R"((
        Returns a dictionary detailing the compile-time and runtime capabilities of the library.

        This includes detected SIMD instruction sets and which BVH layouts support
        conversion and traversal on the current system.

        Returns:
            Dict[str, Any]: A dictionary with the hardware info.
        ))");

    m.def("layout_to_string", [](Layout L){ return std::string(layout_to_string(L)); });

    m.def("supports_layout",
        [](Layout L, bool for_traversal) { return supports_layout(L, for_traversal); },
        R"((
        Checks if the current system supports a given BVH layout.

        Args:
            layout (Layout): The layout to check.
            for_traversal (bool): If True (default), checks if the layout is supported for
                                  ray traversal. If False, checks if it's supported for
                                  conversion (which is always True for valid layouts).

        Returns:
            bool: True if the layout is supported, False otherwise.
        ))",
        nb::arg("layout"), nb::arg("for_traversal") = true);

    m.def("require_layout",
        [](Layout L, bool for_traversal) {
            if (!supports_layout(L, for_traversal)) {
              throw std::runtime_error(
                  std::string("Requested layout '") + layout_to_string(L) + "' unavailable: " +
                  explain_requirement(L));
            }
        },
        R"((
        Asserts that a given BVH layout is supported, raising a RuntimeError if not.

        This is useful for writing tests or code that depends on a specific high-performance
        layout being available.

        Args:
            layout (Layout): The layout to require.
            for_traversal (bool): If True (default), requires traversal support.

        Raises:
            RuntimeError: If the layout is not supported on the current system.
        ))",
        nb::arg("layout"), nb::arg("for_traversal") = true);

    // Main Python classes
    nb::class_<PyRay>(m, "Ray", "Represents a ray for intersection queries.")

        .def("__init__", [](PyRay *self, const tinybvh::bvhvec3& origin, const tinybvh::bvhvec3& direction, float t, uint32_t mask) {
           new (self) PyRay{
               origin,
               direction,
               t,
               0.0f, // u
               0.0f, // v
               static_cast<uint32_t>(-1), // prim_id
               static_cast<uint32_t>(-1), // inst_id
               mask
           };
        },
        nb::arg("origin"), nb::arg("direction"), nb::arg("t") = 1e30f, nb::arg("mask") = 0xFFFF)

        .def_prop_rw("origin",
            [](const PyRay &r) { return nb::make_tuple(r.origin.x, r.origin.y, r.origin.z); },
            [](PyRay &r, const tinybvh::bvhvec3& orig) { r.origin = orig; },
            "The origin point of the ray (list, tuple, or numpy array).")

        .def_prop_rw("direction",
            [](const PyRay &r) { return nb::make_tuple(r.direction.x, r.direction.y, r.direction.z); },
            [](PyRay &r, const tinybvh::bvhvec3& dir) { r.direction = dir; },
            "The direction vector of the ray (list, tuple, or numpy array).")

        .def_rw("t", &PyRay::t, "The maximum distance for intersection. Updated with hit distance.")

        .def_rw("mask", &PyRay::mask, "The visibility mask for the ray.")

        .def_ro("u", &PyRay::u, "Barycentric u-coordinate, or texture coordinate u when using custom geometry.")

        .def_ro("v", &PyRay::v, "Barycentric v-coordinate, or texture coordinate v when using custom geometry.")

        .def_ro("prim_id", &PyRay::prim_id, "The ID of the primitive hit (-1 for miss).")

        .def_ro("inst_id", &PyRay::inst_id, "The ID of the instance hit (-1 for miss or BLAS hit).")

        .def("__repr__", [](const PyRay &r) {
            std::string origin_s = "[" + std::to_string(r.origin.x) + ", " +
                                         std::to_string(r.origin.y) + ", " +
                                         std::to_string(r.origin.z) + "]";
            std::string dir_s = "[" + std::to_string(r.direction.x) + ", " +
                                      std::to_string(r.direction.y) + ", " +
                                      std::to_string(r.direction.z) + "]";

            if (r.prim_id != static_cast<uint32_t>(-1)) {
                return "<pytinybvh.Ray (origin=" + origin_s + " dir=" + dir_s +
                       ", Hit inst " + std::to_string(r.inst_id) +
                       " prim " + std::to_string(r.prim_id) +
                       " at t=" + std::to_string(r.t) + ")>";
            }
            return "<pytinybvh.Ray (origin=" + origin_s + " dir=" + dir_s + ", Miss)>";
        });


    nb::class_<PyBVH>(m, "BVH", "A Bounding Volume Hierarchy for fast ray intersections.")

        // Core Builders

        .def_static("from_vertices", &PyBVH::from_vertices,
            R"((
            Builds a BVH from a flat array of vertices (N * 3, 4).

            This is a zero-copy operation. The BVH will hold a reference to the
            provided numpy array's memory buffer. The array must not be garbage-collected
            while the BVH is in use. The number of vertices must be a multiple of 3.

            Args:
                vertices (numpy.ndarray): A float32 array of shape (M, 4).
                quality (BuildQuality): The desired quality of the BVH.
                traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
                intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
                hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

            Returns:
                BVH: A new BVH instance.
            ))",
            nb::rv_policy::move,
            nb::arg("vertices").noconvert(), nb::arg("quality") = BuildQuality::Balanced,
            nb::arg("traversal_cost") = C_TRAV, nb::arg("intersection_cost") = C_INT,
            nb::arg("hq_bins") = HQBVHBINS,
            nb::keep_alive<0,1>()  // result keeps vertices
            )

        .def_static("from_indexed_mesh", &PyBVH::from_indexed_mesh,
            R"((
            Builds a BVH from a vertex buffer and an index buffer.

            This is the most memory-efficient method for triangle meshes and allows for
            efficient refitting after vertex deformation. This is a zero-copy operation.
            The BVH will hold a reference to both provided numpy arrays.

            Args:
                vertices (numpy.ndarray): A float32 array of shape (V, 4), where V is the number of unique vertices.
                indices (numpy.ndarray): A uint32 array of shape (N, 3), where N is the number of triangles.
                quality (BuildQuality): The desired quality of the BVH.
                traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
                intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
                hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

            Returns:
                BVH: A new BVH instance.
            ))",
            nb::rv_policy::move,
            nb::arg("vertices").noconvert(), nb::arg("indices").noconvert(),
            nb::arg("quality") = BuildQuality::Balanced,
            nb::arg("traversal_cost") = C_TRAV, nb::arg("intersection_cost") = C_INT,
            nb::arg("hq_bins") = HQBVHBINS,
            nb::keep_alive<0,1>(),  // result keeps vertices
            nb::keep_alive<0,2>()   // result keeps indices
            )

        .def_static("from_aabbs", &PyBVH::from_aabbs,
            R"((
            Builds a BVH from an array of Axis-Aligned Bounding Boxes.

            This is a zero-copy operation. The BVH will hold a reference to the
            provided numpy array's memory buffer. This is useful for building a BVH over
            custom geometry or for creating a Top-Level Acceleration Structure (TLAS).

            Args:
                aabbs (numpy.ndarray): A float32, C-contiguous array of shape (N, 2, 3),
                                       where each item is a pair of [min_corner, max_corner].
                quality (BuildQuality): The desired quality of the BVH.
                traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
                intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
                hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

            Returns:
                BVH: A new BVH instance.
            ))",
            nb::rv_policy::move,
            nb::arg("aabbs").noconvert(), nb::arg("quality") = BuildQuality::Balanced,
            nb::arg("traversal_cost") = C_TRAV, nb::arg("intersection_cost") = C_INT,
            nb::arg("hq_bins") = HQBVHBINS,
            nb::keep_alive<0,1>()  // result keeps aabbs
            )

        // Convenience builders (with copying)

        .def_static("from_triangles", &PyBVH::from_triangles,
            R"((
            Builds a BVH from a standard triangle array. This is a convenience method that
            copies and reformats the data into the layout required by the BVH.

            Args:
                triangles (numpy.ndarray): A float32 array of shape (N, 3, 3) or (N, 9) representing N triangles.
                quality (BuildQuality): The desired quality of the BVH.
                traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
                intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
                hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

            Returns:
                BVH: A new BVH instance.
            ))",
            nb::rv_policy::move,
            nb::arg("triangles"), nb::arg("quality") = BuildQuality::Balanced,
            nb::arg("traversal_cost") = C_TRAV, nb::arg("intersection_cost") = C_INT,
            nb::arg("hq_bins") = HQBVHBINS,
            nb::keep_alive<0,1>()  // result keeps triangles
            )

        .def_static("from_points", &PyBVH::from_points,
            R"((
            Builds a BVH from a point cloud. This is a convenience method that creates an
            axis-aligned bounding box for each point and builds the BVH from those.

            Args:
                points (numpy.ndarray): A float32 array of shape (N, 3) representing N points.
                radius (float): The radius used to create an AABB for each point.
                quality (BuildQuality): The desired quality of the BVH.
                traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
                intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
                hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

            Returns:
                BVH: A new BVH instance.
            ))",
            nb::rv_policy::move,
            nb::arg("points"), nb::arg("radius") = 1e-5f, nb::arg("quality") = BuildQuality::Balanced,
            nb::arg("traversal_cost") = C_TRAV, nb::arg("intersection_cost") = C_INT,
            nb::arg("hq_bins") = HQBVHBINS,
            nb::keep_alive<0,1>()  // result keeps points
            )

        // Cache policy management

        .def("set_cache_policy", &PyBVH::set_cache_policy,
            R"((
            Sets caching policy for converted layouts.

            Args:
                policy (CachePolicy): The new policy to use (ActiveOnly or All).
            ))",
           nb::arg("policy"))

        .def("clear_cached_layouts", &PyBVH::clear_cached_layouts,
           R"((
            Frees the memory of all cached layouts, except for the active one and the base layout.
            ))")

        // Conversion, TLAS building

        .def("convert_to", &PyBVH::convert_to,
            R"((
            Converts the BVH to a different internal memory layout, modifying it in-place.

            This allows optimizing the BVH for different traversal algorithms (SSE, AVX, etc.).
            The caching policy of converted layouts can be controlled (see `set_cache_policy` and `clear_cached_layouts`).

            Args:
                layout (Layout): The target memory layout.
                compact (bool): Whether to compact the BVH during conversion. Defaults to True.
                strict (bool): If True, raises a RuntimeError if the target layout is not
                               supported for traversal on the current system. Defaults to False.
            ))",
           nb::arg("layout") = Layout::Standard,
           nb::arg("compact") = true,
           nb::arg("strict") = false)

        .def_static("build_tlas", &PyBVH::build_tlas,
            R"((
            Builds a Top-Level Acceleration Structure (TLAS) from a list of BVH instances.

            Args:
                instances (numpy.ndarray): A structured array with `instance_dtype` describing
                                           each instance's transform, blas_id, and mask.
                BLASes (List[BVH]): A list of the BVH objects to be instanced. The `blas_id`
                                    in the instances array corresponds to the index in this list.

            Returns:
                BVH: A new BVH instance representing the TLAS.
            ))",
            nb::arg("instances"), nb::arg("BLASes"))

        // Intersection methods

        .def("intersect", &PyBVH::intersect,
            R"((
            Performs an intersection query with a single ray.

            This method modifies the passed Ray object in-place if a closer hit is found.

            Args:
                ray (Ray): The ray to test. Its `t`, `u`, `v`, and `prim_id` attributes
                           will be updated upon a successful hit.

            Returns:
                float: The hit distance `t` if a hit was found, otherwise `infinity`.
            ))",
            nb::arg("ray"))

        .def("intersect_batch", &PyBVH::intersect_batch,
            R"((
            Performs intersection queries for a batch of rays.

            This method is highly parallelized using multi-core processing for all geometry types
            (triangles, AABBs, spheres). For standard triangle meshes, it also leverages SIMD
            instructions where available for maximum throughput.

            .. warning::
                Packet traversal (`Auto` or `Force`) is highly optimized but
                makes strict assumptions about the input rays:
                    - Shared origin: All rays in a batch must share the same origin point.
                    - Coherent directions: The ray directions should form a coherent frustum,
                        like rays cast from a camera through pixels on a screen.

                Providing rays with different origins or random, incoherent directions
                with packet traversal enabled may lead to **incorrect results (false misses)**.
                If your rays do not meet these criteria, use `packet='Never'` to ensure
                correctness via scalar traversal.

            Args:
                origins (numpy.ndarray): A (N, 3) float array of ray origins.
                directions (numpy.ndarray): A (N, 3) float array of ray directions.
                t_max (numpy.ndarray, optional): A (N,) float array of maximum intersection distances.
                masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                                 For a ray to test an instance for intersection, the bitwise
                                                 AND of the ray's mask and the instance's mask must be non-zero.
                                                 If not provided, rays default to mask 0xFFFF (intersect all instances).
                packet (PacketMode, optional): Choose packet usage strategy. Defaults to Auto.
                same_origin_eps (float, optional): Epsilon for same-origin test. Default 1e-6.
                max_spread (float, optional): Max spread allowed for a batch (cone angle, in degrees). Default 1.0.
                warn_on_incoherent (bool, optional): Warn when rays differ in origin. Default True.

            Returns:
                numpy.ndarray: A structured array of shape (N,) with dtype
                    [('prim_id', '<u4'), ('inst_id', '<u4'), ('t', '<f4'), ('u', '<f4'), ('v', '<f4')].
                    For misses, prim_id and inst_id are -1 and t is infinity.
                    For TLAS hits, inst_id is the instance index and prim_id is the primitive
                    index within that instance's BLAS.
            ))",
            nb::arg("origins").noconvert(), nb::arg("directions").noconvert(),
            nb::arg("t_max").noconvert() = nb::none(), nb::arg("masks").noconvert() = nb::none(),
            nb::arg("packet") = PacketMode::Auto,
            nb::arg("same_origin_eps") = 1e-6f,
            nb::arg("max_spread") = 1.0f,
            nb::arg("warn_on_incoherent") = true)

        .def("is_occluded", &PyBVH::is_occluded,
            R"((
            Performs an occlusion query with a single ray.

            Checks if any geometry is hit by the ray within the distance specified by `ray.t`.
            This is typically faster than `intersect` as it can stop at the first hit.

            Args:
                ray (Ray): The ray to test.

            Returns:
                bool: True if the ray is occluded, False otherwise.
           ))",
           nb::arg("ray"))

        .def("is_occluded_batch", &PyBVH::is_occluded_batch,
            R"((
            Performs occlusion queries for a batch of rays, parallelized for performance.

            .. warning::
                Packet traversal (`Auto` or `Force`) is highly optimized but
                makes strict assumptions about the input rays:
                    - Shared origin: All rays in a batch must share the same origin point.
                    - Coherent directions: The ray directions should form a coherent frustum,
                        like rays cast from a camera through pixels on a screen.

                Providing rays with different origins or random, incoherent directions
                with packet traversal enabled may lead to **incorrect results (false misses)**.
                If your rays do not meet these criteria, use `packet='Never'` to ensure
                correctness via scalar traversal.

            Args:
                origins (numpy.ndarray): A (N, 3) float array of ray origins.
                directions (numpy.ndarray): A (N, 3) float array of ray directions.
                t_max (numpy.ndarray, optional): A (N,) float array of maximum occlusion distances.
                                                 If a hit is found beyond this distance, it is ignored.
                masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                                 For a ray to test an instance for intersection, the bitwise
                                                 AND of the ray's mask and the instance's mask must be non-zero.
                                                 If not provided, rays default to mask 0xFFFF (intersect all instances).
                packet (PacketMode, optional): Choose packet usage strategy. Defaults to Auto.
                same_origin_eps (float, optional): Epsilon for same-origin test. Default 1e-6.
                max_spread (float, optional): Max spread allowed for a batch (cone angle, in degrees). Default 1.0.
                warn_on_incoherent (bool, optional): Warn when rays differ in origin. Default True.

            Returns:
                numpy.ndarray: A boolean array of shape (N,) where `True` indicates occlusion.
           ))",
            nb::arg("origins").noconvert(), nb::arg("directions").noconvert(),
            nb::arg("t_max").noconvert() = nb::none(), nb::arg("masks").noconvert() = nb::none(),
            nb::arg("packet") = PacketMode::Auto,
            nb::arg("same_origin_eps") = 1e-6f,
            nb::arg("max_spread") = 1.0f,
            nb::arg("warn_on_incoherent") = true)

        .def("intersect_sphere", &PyBVH::intersect_sphere,
            R"((
            Checks if any geometry intersects with a given sphere.

            This is useful for proximity queries or collision detection. It stops at the
            first intersection found. Note: This method is not implemented for custom
            geometry (AABBs, points) and will only work on triangle meshes.

            Args:
                center (Vec3Like): The center of the sphere.
                radius (float): The radius of the sphere.

            Returns:
                bool: True if an intersection is found, False otherwise.
            ))",
            nb::arg("center"), nb::arg("radius"))

        // Accessors for the BVH data

        .def_prop_ro("nodes", &PyBVH::get_nodes,
            "The structured numpy array of BVH nodes (only for standard layout).")

        .def_prop_ro("prim_indices", &PyBVH::get_prims_indices,
            "The BVH-ordered array of primitive indices (only for standard layout).")

        // Accessors for source geometry

        .def_prop_ro("source_vertices", [](const PyBVH& self) -> nb::object {
            if (self.custom_type == PyBVH::CustomType::None && !self.source_geometry_refs.empty()) {
                return self.source_geometry_refs[0];
            }
            return nb::none();
            },
            "The source vertex buffer as a numpy array, or None.")

        .def_prop_ro("source_indices", [](const PyBVH& self) -> nb::object {
            if (self.custom_type == PyBVH::CustomType::None && self.source_geometry_refs.size() > 1) {
                return self.source_geometry_refs[1];
            }
            return nb::none();
            }, "The source index buffer for an indexed mesh as a numpy array, or None.")

        .def_prop_ro("source_aabbs", [](const PyBVH& self) -> nb::object {
            if (self.custom_type == PyBVH::CustomType::AABB && !self.source_geometry_refs.empty()) {
                return self.source_geometry_refs[0];
            }
            return nb::none();
            }, "The source AABB buffer as a numpy array, or None.")

        .def_prop_ro("source_points", [](const PyBVH& self) -> nb::object {
            if (self.custom_type == PyBVH::CustomType::Sphere && !self.source_geometry_refs.empty()) {
                return self.source_geometry_refs[0];
            }
            return nb::none();
            }, "The source point buffer for sphere geometry as a numpy array, or None.")

        .def_prop_ro("sphere_radius", [](const PyBVH& self) -> nb::object {
            if (self.custom_type == PyBVH::CustomType::Sphere) {
                return nb::cast(self.sphere_radius);
            }
            return nb::none();
            }, "The radius for sphere geometry, or None.")

        .def("get_buffers", &PyBVH::get_buffers,
            R"((
            Returns a dictionary of raw, zero-copy numpy arrays for all current BVH's internal data and geometry.

            This provides low-level access to all the underlying C++ buffers. The returned arrays
            are views into the BVH's memory.
            The structure of the returned dictionary and the shape/content of the arrays depend on the active layout.

            This is primarily useful for advanced use cases (sending BVH data to a SSBO, etc.).

            Returns:
                Dict[str, numpy.ndarray]: A dictionary mapping buffer names (e.g., 'nodes',
                                        'prim_indices', 'packed_data', 'vertices', etc.) to
                                        their corresponding raw data arrays.
            ))")

        // Load / save

        .def_static("load", &PyBVH::load,
            R"((
            Loads a BVH from a file, re-linking it to the provided geometry.

            The geometry must be provided in the same layout as when the BVH was originally
            built and saved. If it was built from an indexed mesh, both the vertices and the indices must be provided.

            Args:
                filepath (str or pathlib.Path): The path to the saved BVH file.
                vertices (numpy.ndarray): A float32, C-style contiguous numpy array of shape (V, 4)
                                          representing the vertex data.
                indices (numpy.ndarray, optional): A uint32, C-style contiguous array of shape (N, 3)
                                                   if the BVH was built from an indexed mesh.

            Returns:
                BVH: A new BVH instance.
            ))",
            nb::arg("filepath"), nb::arg("vertices"), nb::arg("indices") = nb::none())

        .def("save", &PyBVH::save,
            R"((
            Saves the BVH to a file.

            Args:
                filepath (str or pathlib.Path): The path where the BVH file will be saved.
            ))",
            nb::arg("filepath"))

        // Advanced manipulation methods

        .def("refit", &PyBVH::refit,
            R"((
            Refits the BVH to the current state of the source geometry, which is much
            faster than a full rebuild.

            Should be called after the underlying vertex data (numpy array used for construction)
            has been modified.

            Note: This will fail if the BVH was built with spatial splits (high-quality preset).
            ))")

        .def("optimize", &PyBVH::optimize,
            R"((
            Optimizes the BVH tree structure to improve query performance.

            This is a costly operation best suited for static scenes. It works by
            re-inserting subtrees into better locations based on the SAH cost.

            Args:
                iterations (int): The number of optimization passes. Defaults to 25.
                extreme (bool): If true, a larger portion of the tree is considered
                                for optimization in each pass. Defaults to False.
                stochastic (bool): If true, uses a randomized approach to select
                                   nodes for re-insertion. Defaults to False.
            ))",
            nb::arg("iterations") = 25, nb::arg("extreme") = false, nb::arg("stochastic") = false)

        .def("compact", &PyBVH::compact,
            R"((
            Removes unused nodes from the BVH structure, reducing memory usage.

            This is useful after building with high quality (which may create
            spatial splits and more primitives) or after optimization, as these
            processes can leave gaps in the node array.
            ))")

        .def("split_leaves", &PyBVH::split_leaves,
            R"((
            Recursively splits leaf nodes until they contain at most `max_prims` primitives.
            This modifies the BVH in-place. Typically used to prepare a BVH for optimization
            by breaking it down into single-primitive leaves.

            Warning: This may fail if the BVH does not have enough pre-allocated node memory.

            Args:
                max_prims (int): The maximum number of primitives a leaf node can contain.
                                 Defaults to 1.
            ))",
            nb::arg("max_prims") = 1)

        .def("combine_leaves", &PyBVH::combine_leaves,
            R"((
            Merges adjacent leaf nodes if doing so improves the SAH cost.
            This modifies the BVH in-place. Typically used as a post-process after
            optimization to create more efficient leaves.

            Warning: This operation makes the internal primitive index array non-contiguous.

            It is highly recommended to call `compact()` after using this method to clean up
            the BVH structure.
            ))")

        .def("set_opacity_maps", &PyBVH::set_opacity_maps,
            R"((
            Sets the opacity micro-maps for alpha testing during intersection.

            The BVH must be built before calling this. The intersection queries will
            automatically use the map to discard hits on transparent parts of triangles.

            Args:
                map_data (numpy.ndarray): A 1D uint32 numpy array containing the packed
                                          bitmasks for all triangles. The size must be
                                          (tri_count * N * N + 31) // 32
                N (int): The resolution of the micro-map per triangle (e.g., 8 for an 8x8 grid).
            ))",
            nb::arg("map_data"), nb::arg("N"))

        // Read-only properties

        .def_prop_ro("traversal_cost",
            [](const PyBVH &self) -> float {
                if (!self.active_bvh_) throw std::runtime_error("BVH is not initialized.");
                return self.active_bvh_->c_trav;
            },
            "The traversal cost used in the Surface Area Heuristic (SAH) calculation during the build.")

        .def_prop_ro("intersection_cost",
            [](const PyBVH &self) -> float {
                if (!self.active_bvh_) throw std::runtime_error("BVH is not initialized.");
                return self.active_bvh_->c_int;
            },
            "The intersection cost used in the Surface Area Heuristic (SAH) calculation during the build.")

        .def_prop_ro("node_count",
            [](const PyBVH &self) -> uint32_t { return self.active_bvh_->usedNodes; },
            "Total number of nodes in the currently active BVH representation.")

        .def_prop_ro("prim_count",
            [](const PyBVH &self) -> uint32_t { return self.active_bvh_->triCount; },
            "Total number of primitives in the BVH.")

        .def_prop_ro("aabb_min",
            [](PyBVH &self) -> nb::ndarray<const float, nb::numpy, nb::c_contig, nb::ro> {
            if (!self.active_bvh_) return {};
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMin data
            auto arr = nb::ndarray<const float, nb::numpy, nb::c_contig, nb::ro>(
                &self.active_bvh_->aabbMin.x,
                { 3 },
                nb::find(&self)
            );
            return arr;
        }, "The minimum corner of the root axis-aligned bounding box.")

        .def_prop_ro("aabb_max",
            [](PyBVH &self) -> nb::ndarray<const float, nb::numpy, nb::c_contig, nb::ro> {
            if (!self.active_bvh_) return {};
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMax data
            auto arr = nb::ndarray<const float, nb::numpy, nb::c_contig, nb::ro>(
                &self.active_bvh_->aabbMax.x,
                { 3 },
                nb::find(&self)
            );
            return arr;
        }, "The maximum corner of the root axis-aligned bounding box.")

        .def_prop_ro("quality",
            [](const PyBVH &self) -> BuildQuality { return self.quality; },
           "The build quality level used to construct the BVH.")

        .def_prop_ro("leaf_count", [](const PyBVH &self) -> int {
            if (!self.active_bvh_) {
                return 0;
            }
            return dispatch_bvh_call(self.active_bvh_, [](auto& bvh) {
                // Use a compile-time if to check if the method exists for the concrete type
                // We know it exists on tinybvh::BVH. It also exists on MBVH<M>
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.LeafCount();
                } else {
                    // Other layouts don't have this method, so returning 0
                    return 0;
                }
            });
            }, "Total number of leaf nodes (only for standard layout).")

        .def_prop_ro("sah_cost", [](const PyBVH &self) -> float {
            if (!self.active_bvh_ || self.active_bvh_->triCount == 0) {
                return INFINITY;
            }
            // Dispatch the call to the appropriate concrete BVH type
            // Calling with SAHCost(0) is compatible with all layouts that have the method
            return dispatch_bvh_call(self.active_bvh_, [](auto& bvh) {
                return bvh.SAHCost(0);
            });
            }, R"((Calculates the Surface Area Heuristic (SAH) cost of the BVH.))")

        .def_prop_ro("epo_cost", [](const PyBVH &self) -> float {
            if (!self.active_bvh_ || self.active_bvh_->triCount == 0) {
                return 0.0f;
            }

            return dispatch_bvh_call(self.active_bvh_, [](auto& bvh) {
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.EPOCost();
                } else {
                    return 0.0f; // EPO is not defined for other layouts
                }
            });
            }, "Calculates the Expected Projected Overlap (EPO) cost of the BVH (only for standard layout).")

        .def_prop_ro("layout", [](const PyBVH &self) -> Layout {
            return self.active_layout_;
            }, "The current active memory layout of the BVH.")

        .def_prop_ro("is_tlas", &PyBVH::is_tlas,
            "Returns True if the BVH is a Top-Level Acceleration Structure (TLAS).")

        .def_prop_ro("is_blas", [](const PyBVH& self) -> bool {
            return !self.is_tlas();
        }, "Returns True if the BVH is a Bottom-Level Acceleration Structure (BLAS).")

        .def_prop_ro("geometry_type", [](const PyBVH &self) -> GeometryType {
            switch (self.custom_type) {
                case PyBVH::CustomType::AABB:   return GeometryType::AABBs;
                case PyBVH::CustomType::Sphere: return GeometryType::Spheres;
                default:                        return GeometryType::Triangles;
            }
        }, "The type of underlying geometry the BVH was built on.")

        .def_prop_ro("is_compact", [](const PyBVH &self) -> bool {
            // may_have_holes is true when it's *not* compact
            return self.active_bvh_ ? !self.active_bvh_->may_have_holes : true;
        }, "Returns True if the BVH is contiguous in memory.")

        .def_prop_ro("is_refittable", [](const PyBVH &self) -> bool {
            if (!self.active_bvh_) return false;
            // refittable status is determined by the base BVH
            return self.base_ ? self.base_->refittable : false;
        }, "Returns True if the BVH can be refitted.")

        .def_prop_ro("cached_layouts", &PyBVH::get_cached_layouts,
            "A list of the BVH layouts currently held in the cache.")

        .def("__repr__", &PyBVH::get_repr)

    ;
}
