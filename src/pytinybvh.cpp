#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define TINYBVH_IMPLEMENTATION
#include "tinybvh/tiny_bvh.h"

#include <vector>
#include <stdexcept>
#include <memory> // for std::unique_ptr
#include <string>
#include <cmath>
#include <cfloat>

#include "capabilities.h"

namespace py = pybind11;    // TODO: replace by nanobind?

// --------------------------- handy things to represent the library's capabilities ---------------------------

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

static inline const char* layout_to_string(Layout L) {
    switch (L) {
        case Layout::Standard: return "Standard";
        case Layout::SoA: return "SoA";
        case Layout::BVH_GPU: return "BVH (GPU)";
        case Layout::MBVH4: return "MBVH4";
        case Layout::MBVH8: return "MBVH8";
        case Layout::BVH4_CPU: return "BVH4 (CPU)";
        case Layout::BVH4_GPU: return "BVH4 (GPU)";
        case Layout::CWBVH: return "BVH8 (CWBVH)";
        case Layout::BVH8_CPU: return "BVH8 (CPU)";
    }
    return "Unknown";
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

// Helper function to convert layout enum to string
std::string layout_to_string(tinybvh::BVHBase::BVHType layout) {
    switch (layout) {
    case tinybvh::BVHBase::LAYOUT_BVH: return "Standard";
    case tinybvh::BVHBase::LAYOUT_BVH_SOA: return "SoA";
    default: return "Unknown";
    }
}

// Helper to extract 3 floats from a Python object (list, tuple, numpy array)
tinybvh::bvhvec3 py_obj_to_vec3(const py::object& obj) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto seq = py::cast<py::sequence>(obj);
        if (seq.size() != 3) {
            throw std::runtime_error("Input sequence must have 3 elements for a 3D vector.");
        }
        return { seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>() };
    }
    if (py::isinstance<py::array>(obj)) {
        auto arr = py::cast<py::array_t<float>>(obj);
        if (arr.ndim() != 1 || arr.size() != 3) {
            throw std::runtime_error("Input numpy array must have shape (3,) for a 3D vector.");
        }
        return { arr.at(0), arr.at(1), arr.at(2) };
    }
    throw std::runtime_error("Input must be a list, tuple, or numpy array of 3 floats.");
}

// Context struct to hold pointers to geometry data for custom callbacks
struct CustomGeometryContext {
    // For AABBs
    const float* aabbs_ptr = nullptr;
    const float* inv_extents_ptr = nullptr;

    // For Spheres
    const float* points_ptr = nullptr;
    float sphere_radius = 0.0f;
};

// This thread_local pointer is only for the duration of the BVH build process for custom geometry
thread_local const float* g_build_aabbs_ptr = nullptr;

// RAII helper to safely set and clear a thread_local pointer during a scoped operation
class TlocalPointerGuard {
public:
    TlocalPointerGuard(const float** tlocal_ptr, const float* data_ptr)
        : ptr_to_tlocal(tlocal_ptr) {
        *ptr_to_tlocal = data_ptr;
    }

    ~TlocalPointerGuard() {
        *ptr_to_tlocal = nullptr;
    }

private:
    const float** ptr_to_tlocal;
};

// C-style callback for building from AABBs
void aabb_build_callback(const unsigned int i, tinybvh::bvhvec3& bmin, tinybvh::bvhvec3& bmax) {
    if (!g_build_aabbs_ptr) throw std::runtime_error("Internal error: AABB data pointer is null in build callback.");
    const size_t offset = i * 6; // 2 vectors * 3 floats
    bmin.x = g_build_aabbs_ptr[offset + 0]; bmin.y = g_build_aabbs_ptr[offset + 1]; bmin.z = g_build_aabbs_ptr[offset + 2];
    bmax.x = g_build_aabbs_ptr[offset + 3]; bmax.y = g_build_aabbs_ptr[offset + 4]; bmax.z = g_build_aabbs_ptr[offset + 5];
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

        constexpr float PI = 3.1415926535f;
        constexpr float INV_PI = 1.0f / PI;
        constexpr float INV_TWO_PI = 1.0f / (2.0f * PI);

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
                           py::ssize_t n, float eps)
{
    if (n <= 1) return true;
    const float ox0 = origins[0 * o_col];
    const float oy0 = origins[1 * o_col];
    const float oz0 = origins[2 * o_col];
    for (py::ssize_t i = 1; i < n; ++i) {
        const size_t base = static_cast<size_t>(i) * o_row;
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

    [[nodiscard]] inline bool cone_ok(py::ssize_t start, int count) const {
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

    static inline bool tmax_ok(py::ssize_t start, int count) {
    // [[nodiscard]] inline bool tmax_ok(py::ssize_t start, int count) {
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


// -------------------------------- internal structs ---------------------------------

// pybind11 wrapper for tinybvh::Ray to use in intersection queries
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

// -------------------------------- these ones are exposed to python ---------------------------------

enum class BuildQuality { Quick, Balanced, High };
enum class GeometryType { Triangles, AABBs, Spheres };
enum class PacketMode { Auto, Never, Force };
enum class CachePolicy : uint8_t { ActiveOnly, All };

// -------------------------------- main pytinybvh classes ---------------------------------

// Helper to dispatch calls to the correct BVH implementation based on its layout
// (the static_cast is safe because we check the 'layout' enum before casting)
template <typename Func>
auto dispatch_bvh_call(tinybvh::BVHBase* bvh_base, Func f) {
    if (!bvh_base) {
        throw std::runtime_error("Cannot dispatch call on a null BVH.");
    }
    // NOLINTBEGIN(cppcoreguidelines-pro-type-static-cast-downcast)
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
            return f(*static_cast<tinybvh::BVH*>(bvh_base)); // fallback for empty BVH

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
    py::list source_geometry_refs;
    std::vector<tinybvh::BLASInstance> instances_data;
    py::array_t<uint32_t> opacity_map_ref;  // ref to keep opacity maps
    std::vector<tinybvh::BVHBase*> blas_pointers; // raw pointers to BLASes for TLAS build

    // Metadata about the BVH that persists across conversions
    BuildQuality quality = BuildQuality::Balanced;
    enum class CustomType { None, AABB, Sphere };
    CustomType custom_type = CustomType::None;
    float sphere_radius = 0.0f;

    // Constructor
    PyBVH() = default;

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

        if (active_layout_ != Layout::SoA)      soa_.reset();

        // Only drop MBVH if nothing currently depends on it
        if (!needs_m4)                           mbvh4_.reset();
        if (!needs_m8)                           mbvh8_.reset();

        if (active_layout_ != Layout::BVH4_CPU)  bvh4_cpu_.reset();
        if (active_layout_ != Layout::BVH4_GPU)  bvh4_gpu_.reset();
        if (active_layout_ != Layout::CWBVH)     cwbvh_.reset();
        if (active_layout_ != Layout::BVH8_CPU)  bvh8_cpu_.reset();
        if (active_layout_ != Layout::BVH_GPU)   bvh_gpu_.reset();
    }

    // Converter method

    void convert_to(Layout target, bool compact=true, bool strict=false) {
        if (!base_) {
            throw std::runtime_error("Cannot convert an uninitialized BVH.");
        }
        // Always allows conversion. Enforces traversal availability only if `strict` is True
        if (strict && !supports_layout(target, /*for_traversal=*/true)) {
            throw std::runtime_error(
                std::string("Conversion to ") + layout_to_string(target) + ": " +
                explain_requirement(target));
        }
        switch (target) {
            case Layout::Standard:
                active_bvh_ = base_.get();
                active_layout_ = Layout::Standard;
                break;

            case Layout::SoA:
                if (!soa_) soa_ = std::make_unique<tinybvh::BVH_SoA>();
                soa_->context = base_->context;
                soa_->ConvertFrom(*base_, compact);
                active_bvh_ = soa_.get();
                active_layout_ = Layout::SoA;
                break;

            case Layout::BVH_GPU:
                if (!bvh_gpu_) bvh_gpu_ = std::make_unique<tinybvh::BVH_GPU>();
                bvh_gpu_->context = base_->context;
                bvh_gpu_->ConvertFrom(*base_, compact);
                active_bvh_ = bvh_gpu_.get();
                active_layout_ = Layout::BVH_GPU;
                break;

            case Layout::MBVH4:
                if (!mbvh4_) {
                    mbvh4_ = std::make_unique<tinybvh::MBVH<4>>();
                    mbvh4_->context = base_->context;
                    mbvh4_->ConvertFrom(*base_, compact);
                }
                active_bvh_ = mbvh4_.get();
                active_layout_ = Layout::MBVH4;
                break;

            case Layout::MBVH8:
                if (!mbvh8_) {
                    mbvh8_ = std::make_unique<tinybvh::MBVH<8>>();
                    mbvh8_->context = base_->context;
                    mbvh8_->ConvertFrom(*base_, compact);
                }
                active_bvh_ = mbvh8_.get();
                active_layout_ = Layout::MBVH8;
                break;

            case Layout::BVH4_CPU:
                if (!mbvh4_) { // lazily create and cache the intermediate MBVH4
                    mbvh4_ = std::make_unique<tinybvh::MBVH<4>>();
                    mbvh4_->context = base_->context;
                    mbvh4_->ConvertFrom(*base_, compact);
                }
                if (!bvh4_cpu_) bvh4_cpu_ = std::make_unique<tinybvh::BVH4_CPU>();
                bvh4_cpu_->context = base_->context;
                bvh4_cpu_->ConvertFrom(*mbvh4_);
                active_bvh_ = bvh4_cpu_.get();
                active_layout_ = Layout::BVH4_CPU;
                break;

            case Layout::BVH4_GPU:
                if (!mbvh4_) { // lazily create and cache the intermediate MBVH4
                    mbvh4_ = std::make_unique<tinybvh::MBVH<4>>();
                    mbvh4_->context = base_->context;
                    mbvh4_->ConvertFrom(*base_, compact);
                }
                if (!bvh4_gpu_) bvh4_gpu_ = std::make_unique<tinybvh::BVH4_GPU>();
                bvh4_gpu_->context = base_->context;
                bvh4_gpu_->ConvertFrom(*mbvh4_, compact);
                active_bvh_ = bvh4_gpu_.get();
                active_layout_ = Layout::BVH4_GPU;
                break;

            case Layout::CWBVH:
                if (!mbvh8_) { // lazily create and cache the intermediate MBVH8
                    mbvh8_ = std::make_unique<tinybvh::MBVH<8>>();
                    mbvh8_->context = base_->context;
                    mbvh8_->ConvertFrom(*base_, compact);
                }
                if (!cwbvh_) cwbvh_ = std::make_unique<tinybvh::BVH8_CWBVH>();
                cwbvh_->context = base_->context;
                cwbvh_->ConvertFrom(*mbvh8_, compact);
                active_bvh_ = cwbvh_.get();
                active_layout_ = Layout::CWBVH;
                break;

            case Layout::BVH8_CPU:
                if (!mbvh8_) { // lazily create and cache the intermediate MBVH8
                    mbvh8_ = std::make_unique<tinybvh::MBVH<8>>();
                    mbvh8_->context = base_->context;
                    mbvh8_->ConvertFrom(*base_, compact);
                }
                if (!bvh8_cpu_) bvh8_cpu_ = std::make_unique<tinybvh::BVH8_CPU>();
                bvh8_cpu_->context = base_->context;
                bvh8_cpu_->ConvertFrom(*mbvh8_);
                active_bvh_ = bvh8_cpu_.get();
                active_layout_ = Layout::BVH8_CPU;
                break;
        }
        if (cache_policy_ == CachePolicy::ActiveOnly) {
            clear_cached_layouts();
        }
    }

    // Helper for builders to finalize the wrapper
    static void finalize_build(const std::unique_ptr<PyBVH>& wrapper, std::unique_ptr<tinybvh::BVH> bvh) {
        wrapper->base_ = std::move(bvh);
        wrapper->active_bvh_ = wrapper->base_.get();
        wrapper->active_layout_ = Layout::Standard;
    }

    // Core builders (zero-copy)

    static std::unique_ptr<PyBVH> from_vertices(py::array_t<float, py::array::c_style> vertices_np,
        BuildQuality quality, float traversal_cost, float intersection_cost) {

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4 || vertices_np.shape(0) % 3 != 0) {
            throw std::runtime_error("Input vertices must be a 2D numpy array with shape (N*3, 4), where N is the number of triangles.");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->source_geometry_refs.append(vertices_np);   // reference to the numpy array
        wrapper->quality = quality;
        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;

        if (vertices_np.shape(0) == 0) {
            finalize_build(wrapper, std::move(bvh));
            return wrapper;
        }

        py::buffer_info vertices_buf = vertices_np.request();
        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
        const auto prim_count = static_cast<uint32_t>(vertices_buf.shape[0] / 3);

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

        finalize_build(wrapper, std::move(bvh)); // transfer ownership
        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_indexed_mesh(
        py::array_t<float, py::array::c_style> vertices_np,
        py::array_t<uint32_t, py::array::c_style> indices_np,
        BuildQuality quality, float traversal_cost, float intersection_cost) {

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input vertices must be a 2D numpy array with shape (V, 4).");
        }
        if (indices_np.ndim() != 2 || indices_np.shape(1) != 3) {
            throw std::runtime_error("Input indices must be a 2D numpy array with shape (N, 3).");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->source_geometry_refs.append(vertices_np);  // references to vertices numpy array
        wrapper->source_geometry_refs.append(indices_np);   // and indexes numpy array
        wrapper->quality = quality;

        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;

        if (vertices_np.shape(0) == 0  || indices_np.shape(0) == 0) {
            finalize_build(wrapper, std::move(bvh));  // transfer ownership
            return wrapper;
        }

        py::buffer_info vertices_buf = vertices_np.request();
        py::buffer_info indices_buf = indices_np.request();

        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
        auto indices_ptr = static_cast<const uint32_t*>(indices_buf.ptr);
        const auto prim_count = static_cast<uint32_t>(indices_buf.shape[0]);

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

        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_aabbs(py::array_t<float, py::array::c_style> aabbs_np,
        BuildQuality quality, float traversal_cost, float intersection_cost) {
        if (aabbs_np.ndim() != 3 || aabbs_np.shape(1) != 2 || aabbs_np.shape(2) != 3) {
            throw std::runtime_error("Input must be a 3D numpy array with shape (N, 2, 3).");
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

        py::buffer_info aabbs_buf = aabbs_np.request();
        const auto* aabbs_ptr = static_cast<const float*>(aabbs_buf.ptr);
        const auto prim_count = static_cast<uint32_t>(aabbs_buf.shape[0]);

        // Pre-compute reciprocal extents for faster intersection tests
        auto inv_extents_np = py::array_t<float>(py::array::ShapeContainer({static_cast<py::ssize_t>(prim_count), 3}));
        float* inv_extents_ptr = inv_extents_np.mutable_data();

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

        // the RAII guard manages the thread_local pointer for the build
        TlocalPointerGuard build_guard(&g_build_aabbs_ptr, aabbs_ptr);

        // Build is now safe

        // BuildQuick and BuildHQ don't support custom AABB getters, so default to Balanced quality
        bvh->Build(aabb_build_callback, prim_count);

        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;

        // `build_guard` goes out of scope, g_build_aabbs_ptr is set back to nullptr
    }

    // Convenience builders

    static std::unique_ptr<PyBVH> from_triangles(const py::array_t<float, py::array::c_style | py::array::forcecast>& tris_np,
        BuildQuality quality, float traversal_cost, float intersection_cost) {

        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }

        const size_t num_tris = tris_np.shape(0);

        auto wrapper = std::make_unique<PyBVH>();
        auto bvh = std::make_unique<tinybvh::BVH>();    // create a temporary BVH

        wrapper->quality = quality;

        bvh->c_trav = traversal_cost;
        bvh->c_int = intersection_cost;

        if (num_tris == 0) {
            finalize_build(wrapper, std::move(bvh));  // transfer ownership
            return wrapper;
        }

        //new numpy array to hold the reformatted data
        auto vertices_np = py::array_t<float>(py::array::ShapeContainer({static_cast<py::ssize_t>(num_tris) * 3, 4}));

        // reference to it *in* the wrapper so it doesn't get garbage collected
        wrapper->source_geometry_refs.append(vertices_np);

        // Reformat the data
        const float* tris_ptr = tris_np.data();
        float* v_ptr = vertices_np.mutable_data();
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

        py::buffer_info vertices_buf = vertices_np.request();
        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
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

        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_points(py::array_t<float, py::array::c_style | py::array::forcecast> points_np,
        float radius, BuildQuality quality, float traversal_cost, float intersection_cost) {

        if (points_np.ndim() != 2 || points_np.shape(1) != 3) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 3).");
        }
        if (radius <= 0.0f) {
            throw std::runtime_error("Point radius must be positive.");
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
        auto aabbs_np = py::array_t<float>(py::array::ShapeContainer({static_cast<py::ssize_t>(num_points), 2, 3}));
        float* aabbs_ptr = aabbs_np.mutable_data();
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
        TlocalPointerGuard build_guard(&g_build_aabbs_ptr, aabbs_ptr);

        bvh->Build(aabb_build_callback, static_cast<uint32_t>(num_points));

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

        tinybvh::Ray ray(py_ray.origin, py_ray.direction, py_ray.t);
        ray.mask = py_ray.mask;

        // Create and populate context locally for this single call
        CustomGeometryContext context;
        if (custom_type == CustomType::AABB) {
            context.aabbs_ptr = py::cast<py::array_t<float>>(source_geometry_refs[0]).data();
            context.inv_extents_ptr = py::cast<py::array_t<float>>(source_geometry_refs[1]).data();
            ray.hit.auxData = &context;
        } else if (custom_type == CustomType::Sphere) {
            context.points_ptr = py::cast<py::array_t<float>>(source_geometry_refs[0]).data();
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

        if (ray.hit.t < py_ray.t) {
            py_ray.t = ray.hit.t; py_ray.u = ray.hit.u; py_ray.v = ray.hit.v;
#if INST_IDX_BITS == 32
            py_ray.prim_id = ray.hit.prim;
            py_ray.inst_id = is_tlas ? ray.hit.inst : static_cast<uint32_t>(-1);
#else
            const uint32_t inst = ray.hit.prim >> INST_IDX_SHFT;
            py_ray.prim_id = ray.hit.prim & PRIM_IDX_MASK;
            py_ray.inst_id = is_tlas ? inst : (uint32_t)-1;
#endif
            return ray.hit.t;
        }
        return INFINITY;
    }

    [[nodiscard]] py::array intersect_batch(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& origins_np,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& directions_np,
        const py::object& t_max_obj, const py::object& masks_obj,
        PacketMode packet = PacketMode::Never,
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
            throw std::runtime_error("Origins must be (N, 3) float32.");
        if (directions_np.ndim() != 2 || directions_np.shape(1) != 3)
            throw std::runtime_error("Directions must be (N, 3) float32.");
        if (origins_np.shape(0) != directions_np.shape(0))
            throw std::runtime_error("Origins and directions must have same N.");

        const py::ssize_t n_rays = origins_np.shape(0);

        const auto o_row = static_cast<size_t>(origins_np.strides(0)) / sizeof(float);
        const auto o_col = static_cast<size_t>(origins_np.strides(1)) / sizeof(float); // should be 1 for (N, 3) C-contiguous

        const auto d_row = static_cast<size_t>(directions_np.strides(0)) / sizeof(float);
        const auto d_col = static_cast<size_t>(directions_np.strides(1)) / sizeof(float); // should be 1 for (N, 3) C-contiguous

        // TODO: Maybe we need (N, 4) actually (for the other layouts)

        if (n_rays == 0)
            return py::array_t<HitRecord>(0);

        // Optional t_max / masks: keep arrays alive across GIL release
        const float* t_max_ptr = nullptr;
        const uint32_t* masks_ptr = nullptr;
        if (!t_max_obj.is_none()) {
            auto t_max_np = py::cast<py::array_t<float>>(t_max_obj);

            if (t_max_np.ndim() != 1 || t_max_np.size() != n_rays)
                throw std::runtime_error("`t_max` must be a 1D float array of length N.");

            t_max_ptr = t_max_np.data();
        }
        if (!masks_obj.is_none()) {
            auto masks_np = py::cast<py::array_t<uint32_t>>(masks_obj);

            if (masks_np.ndim() != 1 || masks_np.size() != n_rays)
                throw std::runtime_error("`masks` must be a 1D uint32 array of length N.");

            masks_ptr = masks_np.data();
        }

        // Custom geometry context: keep source arrays alive across GIL release
        CustomGeometryContext context{};
        if (custom_type == CustomType::AABB) {
            py::array_t<float> aabbs_arr = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            py::array_t<float> inv_extents_arr = py::cast<py::array_t<float>>(source_geometry_refs[1]);

            context.aabbs_ptr = aabbs_arr.data();
            context.inv_extents_ptr = inv_extents_arr.data();

        } else if (custom_type == CustomType::Sphere) {
            py::array_t<float> points_arr = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            context.points_ptr = points_arr.data();
            context.sphere_radius = sphere_radius;
        }

        // Early out: empty BVH
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            py::array_t<HitRecord> result(n_rays);
            auto* rp = result.mutable_data();
            for (py::ssize_t i = 0; i < n_rays; ++i)
                rp[i] = {
                static_cast<uint32_t>(-1),
                static_cast<uint32_t>(-1),
                INFINITY,
                0.0f,
                0.0f
                };
            return result;
        }

        const auto* origins_ptr = origins_np.data();
        const auto* directions_ptr = directions_np.data();

        // Decide base eligibility for packets
        const bool layout_is_std_bvh = (active_bvh_->layout == tinybvh::BVHBase::LAYOUT_BVH);
        bool want_packets = (packet != PacketMode::Never)
                         && layout_is_std_bvh
                         && (custom_type == CustomType::None)
                         && !is_tlas()
                         && (n_rays >= 256);

        // Shared-origin gate
        if (want_packets && !same_origin_ok(origins_ptr, o_row, o_col, n_rays, same_origin_eps)) {
            if (warn_on_incoherent && packet != PacketMode::Never) {
                py::module::import("warnings").attr("warn")(
                    packet == PacketMode::Force ?
                    "pytinybvh: forcing packet traversal, but ray origins differ. "
                    "Packet kernels assume a shared origin and coherent directions; results may be inaccurate."
                    :
                    "pytinybvh: ray origins differ, falling back to scalar traversal. "
                    "Packet traversal also requires coherent directions."
                );
            }
            if (packet == PacketMode::Auto) want_packets = false;
        }

        // Per-block gate: rays cone size + tmax band
        const float tmax_band_ratio = 8.0f; // unused for now anyway
        const float cos_thresh = std::cos(max_cone_deg * static_cast<float>(M_PI) / 180.0f);
        PacketGate gate{ directions_ptr, d_row, d_col, t_max_ptr, cos_thresh, tmax_band_ratio };

        // 64B-aligned ray storage for packet kernels
        // Note: using this buffer even for scalar, cost is negligible
        std::unique_ptr<tinybvh::Ray, void(*)(tinybvh::Ray*)> rays(
            static_cast<tinybvh::Ray*>(tinybvh::malloc64(static_cast<size_t>(n_rays) * sizeof(tinybvh::Ray), nullptr)),
            [](tinybvh::Ray* p){ tinybvh::free64(p, nullptr); }
        );
        if (!rays) throw std::bad_alloc();

        std::vector<HitRecord> out(n_rays);

        {
            py::gil_scoped_release release;     // release GIL to run traversal

            // Construct rays in-place
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (py::ssize_t i = 0; i < n_rays; ++i) {
                const size_t iSo = static_cast<size_t>(i) * o_row;
                const size_t iSd = static_cast<size_t>(i) * d_row;
                const float t_init   = t_max_ptr   ? t_max_ptr[i]   : 1e30f;
                const uint32_t mask  = masks_ptr   ? masks_ptr[i]   : 0xFFFFu;
                new (&rays.get()[i]) tinybvh::Ray(
                    { origins_ptr[iSo + 0 * o_col], origins_ptr[iSo + 1 * o_col], origins_ptr[iSo + 2 * o_col] },
                    { directions_ptr[iSd + 0 * d_col], directions_ptr[iSd + 1 * d_col], directions_ptr[iSd + 2 * d_col] },
                    t_init, mask
                );
                if (custom_type != CustomType::None) {
                    rays.get()[i].hit.auxData = &context;
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
                        for (py::ssize_t i = 0; i < n_rays; i += 256) {
                            const py::ssize_t end = std::min(i + 256, n_rays);
                            if (end - i == 256 && want_packets && gate.cone_ok(i, 256) && gate.tmax_ok(i, 256)) {
                            #if defined(BVH_USEAVX) && defined(BVH_USESSE)
                                bvh.Intersect256RaysSSE(&rays.get()[i]);  // uses aligned SIMD loads and same-origin assumption
                            #else
                                bvh.Intersect256Rays(&rays.get()[i]);     // also assumes same-origin
                            #endif
                            } else {
                                // remining rays: scalar
                                for (py::ssize_t j = i; j < end; ++j) bvh.Intersect(rays.get()[j]);
                            }
                        }
                    } else {
                        // Scalar fallback (works for any rays)
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; ++i) bvh.Intersect(rays.get()[i]);
                    }
                } else {
                    // Other layouts (SoA, etc): scalar path
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(dynamic)
                    #endif
                    for (py::ssize_t i = 0; i < n_rays; ++i) bvh.Intersect(rays.get()[i]);
                }
            });
        } // GIL re-acquired

        // Determine TLAS vs BLAS (only standard BVH exposes isTLAS())
        bool is_tlas = false;
        dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
            if constexpr (std::is_same_v<std::decay_t<decltype(bvh)>, tinybvh::BVH>)
                is_tlas = bvh.isTLAS();
        });

        // Fill POD output
        for (py::ssize_t i = 0; i < n_rays; ++i) {
            const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
            if (rays.get()[i].hit.t < t_init) {
            #if INST_IDX_BITS == 32
                const uint32_t prim = rays.get()[i].hit.prim;
                const uint32_t inst = is_tlas ? rays.get()[i].hit.inst : static_cast<uint32_t>(-1);
            #else
                const uint32_t prim = rays.get()[i].hit.prim & PRIM_IDX_MASK;
                const uint32_t inst = is_tlas ? (rays.get()[i].hit.prim >> INST_IDX_SHFT) : (uint32_t)-1;
            #endif
                out[i] = { prim, inst, rays.get()[i].hit.t, rays.get()[i].hit.u, rays.get()[i].hit.v };
            } else {
                out[i] = { static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), INFINITY, 0.0f, 0.0f };
            }
        }

        // Materialize numpy array and copy once
        py::array_t<HitRecord> result(n_rays);
        std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(HitRecord));
        return result;
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
            py_ray.t);  // The ray's t is used as the maximum distance for the occlusion check

        ray.mask = py_ray.mask;

        // Create and populate context locally for this single call
        CustomGeometryContext context;
        if (custom_type == CustomType::AABB) {
            context.aabbs_ptr = py::cast<py::array_t<float>>(source_geometry_refs[0]).data();
            context.inv_extents_ptr = py::cast<py::array_t<float>>(source_geometry_refs[1]).data();
            ray.hit.auxData = &context;
        } else if (custom_type == CustomType::Sphere) {
            context.points_ptr = py::cast<py::array_t<float>>(source_geometry_refs[0]).data();
            context.sphere_radius = sphere_radius;
            ray.hit.auxData = &context;
        }

        return dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
            return bvh.IsOccluded(ray);
        });
    }

    [[nodiscard]] py::array_t<bool> is_occluded_batch(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& origins_np,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& directions_np,
        const py::object& t_max_obj, const py::object& masks_obj,
        PacketMode packet = PacketMode::Never,
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
            throw std::runtime_error("Origins must be (N, 3) float32.");
        if (directions_np.ndim() != 2 || directions_np.shape(1) != 3)
            throw std::runtime_error("Directions must be (N, 3) float32.");
        if (origins_np.shape(0) != directions_np.shape(0))
            throw std::runtime_error("Origins and directions must have same N.");

        const py::ssize_t n_rays = origins_np.shape(0);

        const auto o_row = static_cast<size_t>(origins_np.strides(0)) / sizeof(float);
        const auto o_col = static_cast<size_t>(origins_np.strides(1)) / sizeof(float);   // should be 1 for (N, 3) C-contiguous

        const auto d_row = static_cast<size_t>(directions_np.strides(0)) / sizeof(float);
        const auto d_col = static_cast<size_t>(directions_np.strides(1)) / sizeof(float); // should be 1 for (N, 3) C-contiguous

        py::array_t<bool> result(n_rays);
        if (n_rays == 0)
            return result;

        // Optional t_max / masks: keep arrays alive across GIL release
        const float* t_max_ptr = nullptr;
        const uint32_t* masks_ptr = nullptr;
        if (!t_max_obj.is_none()) {
            auto t_max_np = py::cast<py::array_t<float>>(t_max_obj);

            if (t_max_np.ndim() != 1 || t_max_np.size() != n_rays)
                throw std::runtime_error("t_max must be a 1D float array of length N.");
            t_max_ptr = t_max_np.data();
        }
        if (!masks_obj.is_none()) {
            auto masks_np = py::cast<py::array_t<uint32_t>>(masks_obj);

            if (masks_np.ndim() != 1 || masks_np.size() != n_rays)
                throw std::runtime_error("masks must be a 1D uint32 array of length N.");
            masks_ptr = masks_np.data();
        }

        // Custom geometry context: keep source arrays alive across GIL release
        CustomGeometryContext context{};
        if (custom_type == CustomType::AABB) {
            auto aabbs = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            auto invex = py::cast<py::array_t<float>>(source_geometry_refs[1]);
            context.aabbs_ptr = aabbs.data();
            context.inv_extents_ptr = invex.data();

        } else if (custom_type == CustomType::Sphere) {
            auto points = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            context.points_ptr = points.data();
            context.sphere_radius = sphere_radius;
        }

        // Early out: empty BVH
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            auto* outp = result.mutable_data();
            for (py::ssize_t i = 0; i < n_rays; ++i) outp[i] = false;
            return result;
        }

        const auto* origins_ptr = origins_np.data();
        const auto* directions_ptr = directions_np.data();

        // Decide base eligibility for packets
        const bool layout_is_std_bvh = (active_bvh_->layout == tinybvh::BVHBase::LAYOUT_BVH);
        bool want_packets = (packet != PacketMode::Never)
                         && layout_is_std_bvh
                         && (custom_type == CustomType::None)
                         && !is_tlas()
                         && (n_rays >= 256);

        // Shared-origin gate
        if (want_packets && !same_origin_ok(origins_ptr, o_row, o_col, n_rays, same_origin_eps)) {
            if (warn_on_incoherent && packet != PacketMode::Never) {
                py::module::import("warnings").attr("warn")(
                    packet == PacketMode::Force ?
                    "pytinybvh: forcing packet traversal, but ray origins differ. "
                    "Packet kernels assume a shared origin and coherent directions; results may be inaccurate."
                    :
                    "pytinybvh: ray origins differ, falling back to scalar traversal. "
                    "Packet traversal also requires coherent directions."
                );
            }
            if (packet == PacketMode::Auto) want_packets = false;
        }

        // Per-block gate: rays cone size + tmax band
        const float tmax_band_ratio = 8.0f; // unused for now anyway
        const float cos_thresh = std::cos(max_cone_deg * static_cast<float>(M_PI) / 180.0f);
        PacketGate gate{ directions_ptr, d_row, d_col, t_max_ptr, cos_thresh, tmax_band_ratio };

        // Aligned Ray storage
        std::unique_ptr<tinybvh::Ray, void(*)(tinybvh::Ray*)> rays(
            static_cast<tinybvh::Ray*>(tinybvh::malloc64(static_cast<size_t>(n_rays) * sizeof(tinybvh::Ray), nullptr)),
            [](tinybvh::Ray* p){ tinybvh::free64(p, nullptr); }
        );
        if (!rays) throw std::bad_alloc();

        // Output buffer (0/1), written inside traversal
        std::vector<uint8_t> occluded(static_cast<size_t>(n_rays), 0);
        {
            py::gil_scoped_release release;     // release GIL for traversal

            // Construct rays
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (py::ssize_t i = 0; i < n_rays; ++i) {
                const size_t iSo = static_cast<size_t>(i) * o_row;
                const size_t iSd = static_cast<size_t>(i) * d_row;
                const float t_init  = t_max_ptr ? t_max_ptr[i] : 1e30f;
                const uint32_t mask = masks_ptr ? masks_ptr[i] : 0xFFFFu;
                new (&rays.get()[i]) tinybvh::Ray(
                    { origins_ptr[iSo + 0 * o_col], origins_ptr[iSo + 1 * o_col], origins_ptr[iSo + 2 * o_col] },
                    { directions_ptr[iSd + 0 * d_col], directions_ptr[iSd + 1 * d_col], directions_ptr[iSd + 2 * d_col] },
                    t_init, mask
                );
                if (custom_type != CustomType::None) rays.get()[i].hit.auxData = &context;
            }

            dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
                using BVHType = std::decay_t<decltype(bvh)>;

                if constexpr (std::is_same_v<BVHType, tinybvh::BVH>) {  // TODO: Are any other layouts compatible??
                    if (want_packets) {
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; i += 256) {
                            const py::ssize_t end = std::min(i + 256, n_rays);
                            if (end - i == 256 && want_packets && gate.cone_ok(i, 256) && gate.tmax_ok(i, 256)) {
                                // Packet path
                                #if defined(BVH_USEAVX) && defined(BVH_USESSE)
                                bvh.Intersect256RaysSSE(&rays.get()[i]);
                                #else
                                bvh.Intersect256Rays(&rays.get()[i]);
                                #endif
                                // Convert to occlusion flags (t reduced?)
                                for (py::ssize_t j = i; j < end; ++j) {
                                    const float t_init = t_max_ptr ? t_max_ptr[j] : 1e30f;
                                    if (rays.get()[j].hit.t < t_init) occluded[static_cast<size_t>(j)] = 1;
                                }
                            } else {
                                // Fallback: scalar occlusion for this chunk
                                for (py::ssize_t j = i; j < end; ++j) {
                                    if (bvh.IsOccluded(rays.get()[j])) occluded[static_cast<size_t>(j)] = 1;
                                }
                            }
                        }
                    } else {
                        // scalar occlusion
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; ++i) {
                            if (bvh.IsOccluded(rays.get()[i])) occluded[static_cast<size_t>(i)] = 1;
                        }
                    }
                } else {
                    // Other layouts: scalar occlusion only
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(dynamic)
                    #endif
                    for (py::ssize_t i = 0; i < n_rays; ++i) {
                        if (bvh.IsOccluded(rays.get()[i])) occluded[static_cast<size_t>(i)] = 1;
                    }
                }
            });
        } // GIL re-acquired

        // Pack Python bools
        auto* outp = result.mutable_data();
        for (py::ssize_t i = 0; i < n_rays; ++i) outp[i] = (occluded[static_cast<size_t>(i)] != 0);
        return result;
    }

    [[nodiscard]] bool intersect_sphere(const py::object& center_obj, float radius) const {
        if (!active_bvh_ || active_bvh_->triCount == 0) {
            return false;
        }
        if (active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Sphere intersection is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto* bvh = static_cast<const tinybvh::BVH*>(active_bvh_);
        tinybvh::bvhvec3 center = py_obj_to_vec3(center_obj);
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
            throw std::runtime_error("BVH is not refittable. This is expected for a BVH built with high quality (spatial splits).");
        }
        bvh->Refit();
    }

    static std::unique_ptr<PyBVH> build_tlas(py::array instances_np, const py::list& blases_py) {

        // Validation
        if (instances_np.ndim() != 1) {
            throw std::runtime_error("Instances must be a 1D structured numpy array.");
        }
        auto fields = instances_np.dtype().attr("fields").cast<py::dict>();
        if (!fields.contains("transform") || !fields.contains("blas_id")) {
            throw std::runtime_error("Instance dtype must contain 'transform' and 'blas_id'.");
        }
        const py::ssize_t inst_count = instances_np.shape(0);
        if (inst_count == 0) {
            return std::make_unique<PyBVH>();   // return empty BVH
        }

        // Create wrapper that will hold the TLAS
        auto wrapper = std::make_unique<PyBVH>();

        // Prepare data holders
        wrapper->instances_data.resize(inst_count);
        wrapper->blas_pointers.reserve(blases_py.size());

        // Extract the BVHBase* pointers from the BLASes
        for (const auto& blas_obj : blases_py) {
            auto& py_blas = blas_obj.cast<PyBVH&>();
            if (!py_blas.active_bvh_) {
                throw std::runtime_error("One of the provided BLASes is uninitialized.");
            }
            // Get raw pointer from the unique_ptr. This correctly gets the BVHBase*
            // regardless of the BLAS's actual layout (Standard, SoA, etc.)
            wrapper->blas_pointers.push_back(py_blas.active_bvh_);
        }

        // Populate instance data
        for (py::ssize_t i = 0; i < inst_count; ++i) {
            auto record = instances_np[py::int_(i)];
            auto transform_arr = record["transform"].cast<py::array_t<float>>();
            wrapper->instances_data[i].blasIdx = record["blas_id"].cast<uint32_t>();

            if (fields.contains("mask")) {
                wrapper->instances_data[i].mask = record["mask"].cast<uint32_t>();
            }
            std::memcpy(wrapper->instances_data[i].transform.cell, transform_arr.data(), 16 * sizeof(float));
        }

        // Keep Python objects alive to prevent their data from being garbage collected
        wrapper->source_geometry_refs.append(instances_np);
        wrapper->source_geometry_refs.append(blases_py);

        // Build TLAS and assign it to the wrapper
        // (a TLAS is always a standard-layout tinybvh::BVH)
        auto tlas_bvh = std::make_unique<tinybvh::BVH>();

        tlas_bvh->Build(
            wrapper->instances_data.data(),
            static_cast<uint32_t>(inst_count),
            wrapper->blas_pointers.data(),
            static_cast<uint32_t>(wrapper->blas_pointers.size())
        );

        finalize_build(wrapper, std::move(tlas_bvh)); // transfer ownership
        return wrapper;
    }

    // Load / save

    static std::unique_ptr<PyBVH> load(const py::object& filepath_obj, py::array_t<float,
        py::array::c_style | py::array::forcecast> vertices_np, const py::object& indices_obj) { // py::object to accept an array or None

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Vertex data for loading must be a 2D numpy array with shape (V, 4).");
        }

        auto filepath = py::str(filepath_obj).cast<std::string>();
        auto wrapper = std::make_unique<PyBVH>();

        // create a temporary BVH to load into
        auto bvh = std::make_unique<tinybvh::BVH>();
        bool success = false;

        if (indices_obj.is_none()) {

            // Case: non-indexed geometry
            wrapper->source_geometry_refs.append(vertices_np);

            py::buffer_info vertices_buf = vertices_np.request();
            auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
            const auto prim_count = static_cast<uint32_t>(vertices_buf.shape[0] / 3);

            success = bvh->Load(filepath.c_str(), vertices_ptr, prim_count);

        } else {

            // Case: indexed geometry
            auto indices_np = py::cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(indices_obj);
            if (indices_np.ndim() != 2 || indices_np.shape(1) != 3) {
                throw std::runtime_error("Indices must be a 2D numpy array with shape (N, 3).");
            }

            wrapper->source_geometry_refs.append(vertices_np);
            wrapper->source_geometry_refs.append(indices_np);

            py::buffer_info vertices_buf = vertices_np.request();
            py::buffer_info indices_buf = indices_np.request();
            auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
            auto indices_ptr = static_cast<const uint32_t*>(indices_buf.ptr);
            const auto prim_count = static_cast<uint32_t>(indices_buf.shape[0]);
            success = bvh->Load(filepath.c_str(), vertices_ptr, indices_ptr, prim_count);
        }

        if (!success) {
            throw std::runtime_error(
                "Failed to load BVH from file. Check file integrity, version, and that "
                "you provided the correct geometry layout (indexed vs. non-indexed)."
            );
        }

        // Infer quality and set references
        wrapper->quality = bvh->refittable ? BuildQuality::Balanced : BuildQuality::High;

        wrapper->source_geometry_refs.append(vertices_np);
        if (!indices_obj.is_none()) {
             wrapper->source_geometry_refs.append(py::cast<py::array>(indices_obj));
        }

        finalize_build(wrapper, std::move(bvh));  // transfer ownership
        return wrapper;

    }

    void save(const py::object& filepath_obj) const {
        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Saving is only supported for the standard BVH layout.");
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_);

        auto filepath = py::str(filepath_obj).cast<std::string>();
        bvh->Save(filepath.c_str());
    }

    // Advanced manipulation methods

    void optimize(unsigned int iterations, bool extreme, bool stochastic) {

        // TODO: Why does this produce garbage SOMETIMES ?

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

    void set_opacity_maps(const py::array_t<uint32_t, py::array::c_style>& map_data, uint32_t N) {
        if (map_data.ndim() != 1) {
            throw std::runtime_error("Opacity map data must be a 1D uint32 numpy array.");
        }
        // tinybvh expects a specific size: N*N bits per triangle
        // The array should contain (triCount * N * N) / 32 uint32_t values
        const size_t expected_size = (active_bvh_->triCount * N * N + 31) / 32;
        if (static_cast<size_t>(map_data.size()) != expected_size) {
            throw std::runtime_error("Opacity map data has incorrect size for the given N and primitive count.");
        }

        opacity_map_ref = map_data; // Keep reference
        active_bvh_->SetOpacityMicroMaps(opacity_map_ref.mutable_data(), N);
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

    [[nodiscard]] py::array get_nodes() const {
        if (!active_bvh_ || active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            return py::array(py::dtype::of<tinybvh::BVH::BVHNode>(), {0}, {});
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto* bvh = static_cast<const tinybvh::BVH*>(active_bvh_);
        return py::array_t<tinybvh::BVH::BVHNode>(bvh->usedNodes, bvh->bvhNode, py::cast(*this));
    }

    static py::array get_prims_indices(const PyBVH &self) {
        if (!self.active_bvh_ || self.active_bvh_->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            return py::array_t<uint32_t>(0); // return empty uint32 array if not standard
        }
        // This static_cast is safe because the layout has just been checked
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto* bvh = static_cast<const tinybvh::BVH*>(self.active_bvh_);

        if (bvh->idxCount == 0) {
            return py::array_t<uint32_t>(0);
        }
        return py::array_t<uint32_t>(
            bvh->idxCount,
            bvh->primIdx,
            py::cast(self)
        );
    }

    [[nodiscard]] py::dict get_buffers() const {
        if (!active_bvh_) {
            throw std::runtime_error("BVH is not initialized.");
        }
        py::dict buffers;
        const PyBVH& self = *this; // for py::cast owner

        // Get buffers for the BVH structure itself
        dispatch_bvh_call(active_bvh_, [&](auto& bvh) {
            using BvhType = std::decay_t<decltype(bvh)>;

            // Standard layout
            if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                if (bvh.bvhNode) {
                    // exposes nodes as a raw float array. Each is 32 bytes (8 floats)
                    buffers["nodes"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedNodes * 8),
                        reinterpret_cast<const float*>(bvh.bvhNode),
                        py::cast(self)
                    );
                }
                if (bvh.primIdx) {
                    buffers["prim_indices"] = py::array_t<uint32_t>(
                        {static_cast<py::ssize_t>(bvh.idxCount)},
                        bvh.primIdx,
                        py::cast(self)
                    );
                }
            }
            // GPU layouts
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH_GPU>) { // NOLINT(*-branch-clone) Reason: being explicit is better here
                if (bvh.bvhNode) {
                    // Each node is 64 bytes (16 floats)
                    buffers["nodes"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedNodes * 16),
                        reinterpret_cast<const float*>(bvh.bvhNode),
                        py::cast(self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH4_GPU>) {
                if (bvh.bvh4Data) {
                    // usedBlocks is in units of 16 bytes (a bvhvec4). Total floats = usedBlocks * 4
                    buffers["packed_data"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedBlocks * 4),
                        reinterpret_cast<const float*>(bvh.bvh4Data),
                        py::cast(self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH8_CWBVH>) {
                if (bvh.bvh8Data) {
                    // usedBlocks is in units of 16 bytes
                    buffers["nodes"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedBlocks * 4),
                        reinterpret_cast<const float*>(bvh.bvh8Data),
                        py::cast(self)
                    );
                }
                if (bvh.bvh8Tris) {
                    // this determines the size based on compilation flags
                    #ifdef CWBVH_COMPRESSED_TRIS
                        size_t elements_per_tri = 4 * 4; // 4 vec4s
                    #else
                        size_t elements_per_tri = 3 * 4; // 3 vec4s
                    #endif
                    buffers["triangles"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.triCount * elements_per_tri),
                        reinterpret_cast<const float*>(bvh.bvh8Tris),
                        py::cast(self)
                    );
                }
            }
            // CPU SIMD layouts
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH4_CPU>) {
                if (bvh.bvh4Data) {
                    // usedBlocks is in units of 64 bytes (CacheLine). Total floats = usedBlocks * 16
                    buffers["packed_data"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedBlocks * 16),
                        reinterpret_cast<const float*>(bvh.bvh4Data),
                        py::cast(self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH8_CPU>) {
                if (bvh.bvh8Data) {
                    // usedBlocks is in units of 64 bytes (CacheLine). Total floats = usedBlocks * 16
                    buffers["packed_data"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedBlocks * 16),
                        reinterpret_cast<const float*>(bvh.bvh8Data),
                        py::cast(self)
                    );
                }
            }
            // Intermediate layouts
            else if constexpr (std::is_same_v<BvhType, tinybvh::MBVH<4>> || std::is_same_v<BvhType, tinybvh::MBVH<8>>) {
                 if (bvh.mbvhNode) {
                    // MBVH<4> node is 48 bytes (12 floats). MBVH<8> is 64 bytes (16 floats)
                    constexpr size_t node_size_in_floats = sizeof(typename BvhType::MBVHNode) / sizeof(float);
                    buffers["nodes"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedNodes * node_size_in_floats),
                        reinterpret_cast<const float*>(bvh.mbvhNode),
                        py::cast(self)
                    );
                }
            }
            else if constexpr (std::is_same_v<BvhType, tinybvh::BVH_SoA>) {
                if (bvh.bvhNode) {
                    // Each node is 64 bytes (16 floats)
                    buffers["nodes"] = py::array_t<float>(
                        static_cast<py::ssize_t>(bvh.usedNodes * 16),
                        reinterpret_cast<const float*>(bvh.bvhNode),
                        py::cast(self)
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

    [[nodiscard]] py::list get_cached_layouts() const {
        py::list cached;
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
        repr += std::to_string(active_bvh_->usedNodes) + " nodes, ";

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
        repr += "Layout: " + layout_to_string(active_bvh_->layout) + ", ";
        repr += std::string("Status: ") + (active_bvh_->may_have_holes ? "Not compact" : "Compact");
        repr += ")>";
        return repr;
    }
};


// =============================================== pybind11 module =====================================================

// helper function to build the hardware info dict

py::dict get_hardware_info() {
    py::dict result, compile_time, runtime, compile_simd, runtime_simd, compile_layouts;

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
        py::dict row;
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


// Fail fast if the binary was built with an ISA the current CPU/OS can't run
static inline void isa_compat_check() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    RuntimeCapabilities rt = get_runtime_caps();
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

PYBIND11_MODULE(pytinybvh, m) {
    isa_compat_check();

    m.doc() = "Python bindings for the tinybvh library";

    py::enum_<Layout>(m, "Layout")
        .value("Standard",   Layout::Standard)
        .value("SoA",        Layout::SoA)
        .value("BVH_GPU",    Layout::BVH_GPU)
        .value("MBVH4",      Layout::MBVH4)
        .value("MBVH8",      Layout::MBVH8)
        .value("BVH4_CPU",   Layout::BVH4_CPU)
        .value("BVH4_GPU",   Layout::BVH4_GPU)
        .value("CWBVH",      Layout::CWBVH)
        .value("BVH8_CPU",   Layout::BVH8_CPU);

    m.def("hardware_info", &get_hardware_info,
        "Get architecture, compile-time, and runtime hardware capabilities.");

    m.def("layout_to_string", [](Layout L){ return std::string(layout_to_string(L)); });

    m.def("supports_layout",
          [](Layout L, bool for_traversal) { return supports_layout(L, for_traversal); },
          py::arg("layout"), py::arg("for_traversal") = true);

    m.def("require_layout",
          [](Layout L, bool for_traversal) {
              if (!supports_layout(L, for_traversal)) {
                  throw std::runtime_error(
                      std::string("Requested layout '") + layout_to_string(L) + "' unavailable: " +
                      explain_requirement(L));
              }
          },
          py::arg("layout"), py::arg("for_traversal") = true);

    // numpy dtypes for batched hit records, vec3, and BVH node
    PYBIND11_NUMPY_DTYPE(HitRecord, prim_id, inst_id, t, u, v);

    m.attr("hit_record_dtype") = py::dtype::of<HitRecord>();

    PYBIND11_NUMPY_DTYPE(tinybvh::bvhvec3, x, y, z);

    PYBIND11_NUMPY_DTYPE_EX(tinybvh::BVH::BVHNode, aabbMin, "aabb_min", leftFirst, "left_first", aabbMax, "aabb_max", triCount, "prim_count");
    m.attr("bvh_node_dtype") = py::dtype::of<tinybvh::BVH::BVHNode>();

    // TLAS instance struct and its numpy dtype
    struct TLASInstance {
        float transform[16];
        uint32_t blas_id;
        uint32_t mask;
    };
    PYBIND11_NUMPY_DTYPE(TLASInstance, transform, blas_id, mask);
    m.attr("instance_dtype") = py::dtype::of<TLASInstance>();

    // Build quality, Geometry type, BVH Layout, Packet Mode, and Cache Policy enums exposed to Python
    py::enum_<BuildQuality>(m, "BuildQuality", "Enum for selecting BVH build quality.")
        .value("Quick", BuildQuality::Quick, "Fastest build, lower quality queries.")
        .value("Balanced", BuildQuality::Balanced, "Balanced build time and query performance (default).")
        .value("High", BuildQuality::High, "Slowest build (uses spatial splits), highest quality queries.");

    py::enum_<GeometryType>(m, "GeometryType")
        .value("Triangles", GeometryType::Triangles, "The BVH was built over a triangle mesh.")
        .value("AABBs", GeometryType::AABBs, "The BVH was built over custom Axis-Aligned Bounding Boxes.")
        .value("Spheres", GeometryType::Spheres, "The BVH was built over a point cloud with a radius (spheres).");

    py::enum_<CachePolicy>(m, "CachePolicy")
        .value("ActiveOnly", CachePolicy::ActiveOnly)
        .value("All", CachePolicy::All);

    py::enum_<PacketMode>(m, "PacketMode")
        .value("Auto",  PacketMode::Auto,  "Use packets only if rays have a shared origin. Assumes coherent directions.")
        .value("Never", PacketMode::Never, "Always use scalar traversal. Safest for non-coherent rays.")
        .value("Force", PacketMode::Force, "Force packet traversal. This can provide a speedup, but is unsafe if rays are not coherent (even if they have the same origin).");

    // Main Python classes
    py::class_<PyRay>(m, "Ray", "Represents a ray for intersection queries.")

        .def(py::init([](const py::object& origin_obj, const py::object& direction_obj, float t, uint32_t mask) {
             return PyRay{
                py_obj_to_vec3(origin_obj),
                py_obj_to_vec3(direction_obj),
                t,
                 0.f,
                 0.f,
                 static_cast<uint32_t>(-1),
                 static_cast<uint32_t>(-1), mask
             };
        }),
        py::arg("origin"), py::arg("direction"), py::arg("t") = 1e30f, py::arg("mask") = 0xFFFF)

        .def_property("origin",
            [](const PyRay &r) { return py::make_tuple(r.origin.x, r.origin.y, r.origin.z); },
            [](PyRay &r, const py::object& obj) { r.origin = py_obj_to_vec3(obj); },
            "The origin point of the ray (list, tuple, or numpy array).")

        .def_property("direction",
            [](const PyRay &r) { return py::make_tuple(r.direction.x, r.direction.y, r.direction.z); },
            [](PyRay &r, const py::object& obj) { r.direction = py_obj_to_vec3(obj); },
            "The direction vector of the ray (list, tuple, or numpy array).")

        .def_readwrite("t", &PyRay::t, "The maximum distance for intersection. Updated with hit distance.")

        .def_readwrite("mask", &PyRay::mask, "The visibility mask for the ray.")

        .def_readonly("u", &PyRay::u, "Barycentric u-coordinate for triangle hits, or the first texture coordinate for custom geometry like spheres and AABBs.")

        .def_readonly("v", &PyRay::v, "Barycentric v-coordinate for triangle hits, or the second texture coordinate for custom geometry like spheres and AABBs.")

        .def_readonly("prim_id", &PyRay::prim_id, "The ID of the primitive that was hit.")

        .def_readonly("inst_id", &PyRay::inst_id, "The ID of the instance that was hit.")

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


    py::class_<PyBVH>(m, "BVH", "A Bounding Volume Hierarchy for fast ray intersections.")

        // Cache policy management

        .def("set_cache_policy", &PyBVH::set_cache_policy,
            R"((
                Sets caching policy for converted layouts.

                Args:
                    policy (CachePolicy): The new policy to use (ActiveOnly or All).
                ))",
            py::arg("policy"))

        .def("clear_cached_layouts", &PyBVH::clear_cached_layouts,
            R"((
                Frees the memory of all cached layouts, except for the active one and the base layout.
            ))")

        // Converter method

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
            py::arg("layout") = Layout::Standard,
            py::arg("compact") = true,
            py::arg("strict") = false)

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

                Returns:
                    BVH: A new BVH instance.
            ))",
            // noconvert() to enforce direct memory access!
            py::arg("vertices").noconvert(), py::arg("quality") = BuildQuality::Balanced,
            py::arg("traversal_cost") = C_TRAV, py::arg("intersection_cost") = C_INT)

        .def_static("from_indexed_mesh", &PyBVH::from_indexed_mesh,
            R"((
                Builds a BVH from a vertex buffer and an index buffer.

                This is the most memory-efficient method for triangle meshes and allows for
                efficient refitting after vertex deformation. This is a zero-copy operation.
                The BVH will hold a reference to both provided numpy arrays.

                Args:
                    vertices (numpy.ndarray): A float32 array of shape (V, 4), where V is the
                                              number of unique vertices.
                    indices (numpy.ndarray): A uint32 array of shape (N, 3), where N is the
                                             number of triangles.
                    quality (BuildQuality): The desired quality of the BVH.

                Returns:
                    BVH: A new BVH instance.
            ))",
            // noconvert() to enforce direct memory access!
            py::arg("vertices").noconvert(), py::arg("indices").noconvert(),
            py::arg("quality") = BuildQuality::Balanced,
            py::arg("traversal_cost") = C_TRAV, py::arg("intersection_cost") = C_INT)

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

                Returns:
                    BVH: A new BVH instance.
            ))",
            // noconvert() to enforce direct memory access!
            py::arg("aabbs").noconvert(), py::arg("quality") = BuildQuality::Balanced,
            py::arg("traversal_cost") = C_TRAV, py::arg("intersection_cost") = C_INT)

        // Convenience builders (with copying)

        .def_static("from_triangles", &PyBVH::from_triangles,
            R"((
                Builds a BVH from a standard triangle array. This is a convenience method that
                copies and reformats the data into the layout required by the BVH.

                Args:
                    triangles (numpy.ndarray): A float32 array of shape (N, 3, 3) or (N, 9)
                                               representing N triangles.
                    quality (BuildQuality): The desired quality of the BVH.

                Returns:
                    BVH: A new BVH instance.
            ))",
            // no need for noconvert() here, we copy anyway
            py::arg("triangles"), py::arg("quality") = BuildQuality::Balanced,
            py::arg("traversal_cost") = C_TRAV, py::arg("intersection_cost") = C_INT)

        .def_static("from_points", &PyBVH::from_points,
             R"((
                Builds a BVH from a point cloud. This is a convenience method that creates an
                axis-aligned bounding box for each point and builds the BVH from those.

                Args:
                    points (numpy.ndarray): A float32 array of shape (N, 3) representing N points.
                    radius (float): The radius used to create an AABB for each point.
                    quality (BuildQuality): The desired quality of the BVH.

                Returns:
                    BVH: A new BVH instance.
            ))",
            // no need for noconvert() here, we copy anyway
            py::arg("points"), py::arg("radius") = 1e-5f, py::arg("quality") = BuildQuality::Balanced,
            py::arg("traversal_cost") = C_TRAV, py::arg("intersection_cost") = C_INT)

        // Converter method
        .def("convert_to", &PyBVH::convert_to,
            py::arg("layout"),
            py::arg("compact") = true,
            py::arg("strict") = false)

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
            py::arg("ray"))

        .def("intersect_batch", &PyBVH::intersect_batch,
           R"((
                Performs intersection queries for a batch of rays.

                This method is highly parallelized using multi-core processing for all geometry types
                (triangles, AABBs, spheres). For standard triangle meshes, it also leverages SIMD
                instructions where available for maximum throughput.

                Args:
                    origins (numpy.ndarray): A (N, 3) float array of ray origins.
                    directions (numpy.ndarray): A (N, 3) float array of ray directions.
                    t_max (numpy.ndarray, optional): A (N,) float array of maximum intersection distances.
                    masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                                     For a ray to test an instance for intersection, the bitwise
                                                     AND of the ray's mask and the instance's mask must be non-zero.
                                                     If not provided, rays default to mask 0xFFFF (intersect all instances).
                    packet (PacketMode, optional): Choose packet usage strategy. Defaults to Never.
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
            py::arg("origins"), py::arg("directions"),
            py::arg("t_max") = py::none(), py::arg("masks") = py::none(),
            py::arg("packet") = PacketMode::Never,
            py::arg("same_origin_eps") = 1e-6f,
            py::arg("max_spread") = 1.0f,
            py::arg("warn_on_incoherent") = true)

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
           py::arg("ray"))

        .def("is_occluded_batch", &PyBVH::is_occluded_batch,
           R"((
                Performs occlusion queries for a batch of rays, parallelized for performance.

                This method uses multi-core parallelization for all geometry types (triangles, AABBs, spheres).

                Args:
                    origins (numpy.ndarray): A (N, 3) float array of ray origins.
                    directions (numpy.ndarray): A (N, 3) float array of ray directions.
                    t_max (numpy.ndarray, optional): A (N,) float array of maximum occlusion distances.
                                                     If a hit is found beyond this distance, it is ignored.
                    masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                                     For a ray to test an instance for intersection, the bitwise
                                                     AND of the ray's mask and the instance's mask must be non-zero.
                                                     If not provided, rays default to mask 0xFFFF (intersect all instances).
                    packet (PacketMode, optional): Choose packet usage strategy. Defaults to Never.
                    same_origin_eps (float, optional): Epsilon for same-origin test. Default 1e-6.
                    max_spread (float, optional): Max spread allowed for a batch (cone angle, in degrees). Default 1.0.
                    warn_on_incoherent (bool, optional): Warn when rays differ in origin. Default True.

                Returns:
                    numpy.ndarray: A boolean array of shape (N,) where `True` indicates occlusion.
           ))",
            py::arg("origins"), py::arg("directions"),
            py::arg("t_max") = py::none(), py::arg("masks") = py::none(),
            py::arg("packet") = PacketMode::Never,
            py::arg("same_origin_eps") = 1e-6f,
            py::arg("max_spread") = 1.0f,
            py::arg("warn_on_incoherent") = true)

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
            py::arg("center"), py::arg("radius"))

        // Accessors for the BVH data

        .def_property_readonly("nodes", &PyBVH::get_nodes,
            "The structured numpy array of BVH nodes (only for standard layout).")

        .def_property_readonly("prim_indices", &PyBVH::get_prims_indices,
            "The BVH-ordered array of primitive indices (only for standard layout).")

        // Accessors for source geometry

        .def_property_readonly("source_vertices", [](const PyBVH& self) -> py::object {
            if (self.custom_type == PyBVH::CustomType::None && !self.source_geometry_refs.empty()) {
                return self.source_geometry_refs[0];
            }
            return py::none();
            }, "The source vertex buffer as a numpy array, or None.")

        .def_property_readonly("source_indices", [](const PyBVH& self) -> py::object {
            if (self.custom_type == PyBVH::CustomType::None && self.source_geometry_refs.size() > 1) {
                return self.source_geometry_refs[1];
            }
            return py::none();
            }, "The source index buffer for an indexed mesh as a numpy array, or None.")

        .def_property_readonly("source_aabbs", [](const PyBVH& self) -> py::object {
            if (self.custom_type == PyBVH::CustomType::AABB && !self.source_geometry_refs.empty()) {
                return self.source_geometry_refs[0];
            }
            return py::none();
            }, "The source AABB buffer as a numpy array, or None.")

        .def_property_readonly("source_points", [](const PyBVH& self) -> py::object {
            if (self.custom_type == PyBVH::CustomType::Sphere && !self.source_geometry_refs.empty()) {
                return self.source_geometry_refs[0];
            }
            return py::none();
            }, "The source point buffer for sphere geometry as a numpy array, or None.")

        .def_property_readonly("sphere_radius", [](const PyBVH& self) -> py::object {
            if (self.custom_type == PyBVH::CustomType::Sphere) {
                return py::cast(self.sphere_radius);
            }
            return py::none();
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

        // Refitting, TLAS

        .def("refit", &PyBVH::refit,
             R"((
                Refits the BVH to the current state of the source geometry, which is much faster than a full rebuild.

                Should be called after the underlying vertex data (numpy array used for construction)
                has been modified.

                Note: This will fail if the BVH was built with spatial splits (with BuildQuality.High).
             ))")

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

           py::arg("instances").noconvert(), py::arg("BLASes"))

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
            py::arg("filepath"), py::arg("vertices").noconvert(), py::arg("indices").noconvert() = py::none())

        .def("save", &PyBVH::save,
            R"((
                Saves the BVH to a file.

                Args:
                    filepath (str or pathlib.Path): The path where the BVH file will be saved.
            ))",
            py::arg("filepath"))

        // Advanced manipulation methods

        .def("optimize", &PyBVH::optimize,
             R"((
                Optimizes the BVH tree structure to improve query performance.

                This is a costly operation best suited for static scenes. It works by
                re-inserting subtrees into better locations based on the SAH cost.

                Args:
                    iterations (int): The number of optimization passes.
                    extreme (bool): If true, a larger portion of the tree is considered
                                    for optimization in each pass.
                    stochastic (bool): If true, uses a randomized approach to select
                                       nodes for re-insertion.
             ))",
             py::arg("iterations") = 25, py::arg("extreme") = false, py::arg("stochastic") = false)

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
            py::arg("max_prims") = 1)

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
        py::arg("map_data").noconvert(), py::arg("N"))

        // Read-only properties

        .def_property_readonly("traversal_cost",
            [](const PyBVH &self) {
                if (!self.active_bvh_) throw std::runtime_error("BVH is not initialized.");
                return self.active_bvh_->c_trav;
            },
            "The traversal cost used in the Surface Area Heuristic (SAH) calculation during the build.")

        .def_property_readonly("intersection_cost",
            [](const PyBVH &self) {
                if (!self.active_bvh_) throw std::runtime_error("BVH is not initialized.");
                return self.active_bvh_->c_int;
            },
            "The intersection cost used in the Surface Area Heuristic (SAH) calculation during the build.")

        .def_property_readonly("node_count", [](const PyBVH &self){ return self.active_bvh_->usedNodes; }, "Total number of nodes in the BVH.")

        .def_property_readonly("prim_count", [](const PyBVH &self){ return self.active_bvh_->triCount; }, "Total number of primitives in the BVH.")

        .def_property_readonly("aabb_min", [](PyBVH &self) -> py::array {
            if (!self.active_bvh_) return py::array_t<float>(0);
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMin data
            std::vector<py::ssize_t> shape = {3};
            return py::array_t<float>(
                shape,
                &self.active_bvh_->aabbMin.x,   // data pointer
                py::cast(self)          // owner
            );
        }, "The minimum corner of the root axis-aligned bounding box.")

        .def_property_readonly("aabb_max", [](PyBVH &self) -> py::array {
            if (!self.active_bvh_) return py::array_t<float>(0);
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMax data
            std::vector<py::ssize_t> shape = {3};
            return py::array_t<float>(
                shape,
                &self.active_bvh_->aabbMax.x,   // data pointer
                py::cast(self)          // owner
            );
        }, "The maximum corner of the root axis-aligned bounding box.")

        .def_property_readonly("quality", [](const PyBVH &self) { return self.quality; },
           "The build quality level used to construct the BVH.")

        .def_property_readonly("leaf_count", [](const PyBVH &self) {
            if (!self.active_bvh_) return 0;
            return dispatch_bvh_call(self.active_bvh_, [](auto& bvh) {
                // Use a compile-time if to check if the method exists for the concrete type
                // We know it exists on tinybvh::BVH. It also exists on MBVH<M>
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.LeafCount();
                } else {
                    // Other layouts don't have this method, so returning 0  // TODO: Maybe return None?
                    return 0;
                }
            });
            }, "The total number of leaf nodes (only for standard layout).")

        .def_property_readonly("sah_cost", [](const PyBVH &self) {
            if (!self.active_bvh_ || self.active_bvh_->triCount == 0) {
                return INFINITY;
            }
            // Dispatch the call to the appropriate concrete BVH type
            // Calling with SAHCost(0) is compatible with all layouts that have the method
            return dispatch_bvh_call(self.active_bvh_, [](auto& bvh) {
                return bvh.SAHCost(0);
            });
            }, R"((Calculates the Surface Area Heuristic (SAH) cost of the BVH.))")

        .def_property_readonly("epo_cost", [](const PyBVH &self) {
            if (!self.active_bvh_ || self.active_bvh_->triCount == 0) return 0.0f;
            return dispatch_bvh_call(self.active_bvh_, [](auto& bvh) {
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.EPOCost();
                } else {
                    return 0.0f; // EPO is not defined for other layouts
                }
            });
            }, "Calculates the Expected Projected Overlap (EPO) cost of the BVH (only for standard layout).")

        .def_property_readonly("layout", [](const PyBVH &self) {
            return self.active_layout_;
            }, "The current active memory layout of the BVH.")

        .def_property_readonly("is_tlas", &PyBVH::is_tlas,
            "Returns True if the BVH is a Top-Level Acceleration Structure (TLAS).")

        .def_property_readonly("is_blas", [](const PyBVH& self) {
            return !self.is_tlas();
        }, "Returns True if the BVH is a Bottom-Level Acceleration Structure (BLAS).")

        .def_property_readonly("geometry_type", [](const PyBVH &self) {
            switch (self.custom_type) {
                case PyBVH::CustomType::AABB:   return GeometryType::AABBs;
                case PyBVH::CustomType::Sphere: return GeometryType::Spheres;
                default:                        return GeometryType::Triangles;
            }
        }, "The type of underlying geometry the BVH was built on.")

        .def_property_readonly("is_compact", [](const PyBVH &self) {
            // may_have_holes is true when it's *not* compact
            return self.active_bvh_ ? !self.active_bvh_->may_have_holes : true;
        }, "Returns True if the BVH is contiguous in memory.")

        .def_property_readonly("is_refittable", [](const PyBVH &self) {
            if (!self.active_bvh_) return false;
            // refittable status is determined by the base BVH
            return self.base_ ? self.base_->refittable : false;
        }, "Returns True if the BVH can be refitted.")

        .def_property_readonly("cached_layouts", &PyBVH::get_cached_layouts,
            "A list of the BVH layouts currently held in the cache.")

        .def("__repr__", &PyBVH::get_repr)

    ;
}