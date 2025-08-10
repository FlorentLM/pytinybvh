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

namespace py = pybind11;
using namespace pybind11::literals;

// Macro to check for x86 architecture
#if defined(__x86_64__) || defined(_M_X64)
#define PYTINYBVH_IS_X86 1
#else
#define PYTINYBVH_IS_X86 0
#endif

// -------------------------------- some internal helpers and things ---------------------------------

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

// C-style callback for ray-sphere intersection
bool sphere_intersect_callback(tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!ray.hit.auxData) throw std::runtime_error("Internal error: Ray context is null in sphere intersect callback.");

    auto* context = static_cast<const CustomGeometryContext*>(ray.hit.auxData);

    // Pre-computed reciprocals
    constexpr float PI = 3.1415926535f;
    constexpr float INV_PI = 1.0f / PI;
    constexpr float INV_TWO_PI = 1.0f / (2.0f * PI);

    // Get sphere center
    const size_t offset = prim_id * 3;
    const tinybvh::bvhvec3 center = {context->points_ptr[offset], context->points_ptr[offset + 1], context->points_ptr[offset + 2]};
    const float radius_sq = context->sphere_radius * context->sphere_radius;

    // Ray-sphere intersection
    const tinybvh::bvhvec3 oc = ray.O - center;
    const float a = tinybvh_dot(ray.D, ray.D);
    const float b = 2.0f * tinybvh_dot(oc, ray.D);
    const float c = tinybvh_dot(oc, oc) - radius_sq;
    const float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return false;
    } else {
        float t = (-b - sqrt(discriminant)) * 0.5f; // Since a=1.0
        if (t > 1e-6f && t < ray.hit.t) { // Check for valid t range
            ray.hit.t = t;
            ray.hit.prim = prim_id;

            const tinybvh::bvhvec3 hit_point = ray.O + ray.D * t;
            tinybvh::bvhvec3 normal = tinybvh::tinybvh_normalize(hit_point - center);

            // u = longitude, v = latitude
            ray.hit.u = 0.5f + atan2f(normal.z, normal.x) * INV_TWO_PI;
            ray.hit.v = 0.5f - asinf(normal.y) * INV_PI; // asin maps [-PI/2, PI/2] to [0, 1]
            // TODO: alternatively, could use ray.hit.v = acosf(normal.y) / PI; // Maps [0, PI] to [0, 1]

            return true;
        }
    }
    return false;
}

// C-style callback for occlusion testing with spheres
bool sphere_isoccluded_callback(const tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!ray.hit.auxData) throw std::runtime_error("Internal error: Ray context is null in sphere occlusion callback.");

    auto* context = static_cast<const CustomGeometryContext*>(ray.hit.auxData);

    // Get sphere center
    const size_t offset = prim_id * 3;
    const tinybvh::bvhvec3 center = {context->points_ptr[offset], context->points_ptr[offset + 1], context->points_ptr[offset + 2]};
    const float radius_sq = context->sphere_radius * context->sphere_radius;

    const tinybvh::bvhvec3 oc = ray.O - center;
    const float a = tinybvh::tinybvh_dot(ray.D, ray.D);
    const float b = 2.0f * tinybvh::tinybvh_dot(oc, ray.D);
    const float c = tinybvh::tinybvh_dot(oc, oc) - radius_sq;
    const float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    const float t = (-b - sqrt(discriminant)) * 0.5f; // Since a=1.0
    return (t > 1e-6f && t < ray.hit.t);
}

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

struct HitRecord
{
    uint32_t prim_id;
    uint32_t inst_id;
    float t, u, v;
};


// -------------------------------- these ones are exposed to python ---------------------------------

// Enum for build quality selection exposed to Python
enum class BuildQuality {
    Quick,
    Balanced,
    High
};

// Enum for Geometry type exposed to Python
enum class GeometryType {
    Triangles,
    AABBs,
    Spheres
};

// -------------------------------- main pytinybvh classes ---------------------------------

// Custom deleter for std::unique_ptr<BVHBase>
struct BVHDeleter {
    void operator()(tinybvh::BVHBase* ptr) const {
        if (!ptr) return;
        // Use the layout to cast to the correct derived type before deleting
        switch (ptr->layout) {
            case tinybvh::BVHBase::LAYOUT_BVH:
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
                delete static_cast<tinybvh::BVH*>(ptr);
                break;
            case tinybvh::BVHBase::LAYOUT_BVH_SOA:
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
                delete static_cast<tinybvh::BVH_SoA*>(ptr);
                break;
            case tinybvh::BVHBase::UNDEFINED:
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
                delete static_cast<tinybvh::BVH*>(ptr);
                break;
            default:
                assert(false && "Unknown BVH layout in deleter");
                break;
        }
    }
};

// Helper function to dispatch calls based on the BVH layout
template <typename Func>
auto dispatch_bvh_call(tinybvh::BVHBase* bvh_base, Func f) {
    if (!bvh_base) {
        throw std::runtime_error("Cannot dispatch call on a null BVH.");
    }

    // This static_cast is safe because the 'layout' enum is checked before casting
    // We cannot use dynamic_cast as BVHBase has no virtual functions
    switch (bvh_base->layout) {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-static-cast-downcast)
        case tinybvh::BVHBase::LAYOUT_BVH:
            return f(*static_cast<tinybvh::BVH*>(bvh_base));
        case tinybvh::BVHBase::LAYOUT_BVH_SOA:
            return f(*static_cast<tinybvh::BVH_SoA*>(bvh_base));
        case tinybvh::BVHBase::UNDEFINED:
            return f(*static_cast<tinybvh::BVH*>(bvh_base));
        default:
            throw std::runtime_error("Operation is not supported for this BVH layout.");
        // NOLINTEND(cppcoreguidelines-pro-type-static-cast-downcast)
    }
}

// C++ class to wrap the tinybvh::BVH object and its associated data: this is the object held by the Python BVH class
struct PyBVH {

    // shared_ptr to the standard-layout BVH
    // This will *always* exist and owns the core data arrays (nodes, primIdx)
    std::shared_ptr<tinybvh::BVH> source_bvh;

    // pointer to the *currently active* BVH representation
    // This is a non-owning view. It can point to source_bvh or to an advanced layout.
    tinybvh::BVHBase* active_bvh_view = nullptr;

    // unique_ptr to own the advanced layout object, if one exists
    // If this is null, we are using the standard layout
    std::unique_ptr<tinybvh::BVHBase, BVHDeleter> advanced_layout_ptr;

    // These references must be kept alive for the lifetime of any BVH representation
    py::list source_geometry_refs;
    std::vector<tinybvh::BLASInstance> instances_data;
    py::array_t<uint32_t> opacity_map_ref;  // ref to keep opacity maps
    std::vector<tinybvh::BVHBase*> blas_pointers; // TLAS

    // Metadata about the BVH that persists across conversions
    BuildQuality quality = BuildQuality::Balanced;
    enum class CustomType { None, AABB, Sphere };
    CustomType custom_type = CustomType::None;
    float sphere_radius = 0.0f;

    // Constructor
    PyBVH() {
        source_bvh = std::make_shared<tinybvh::BVH>();
        active_bvh_view = source_bvh.get(); // Initially, we view the source BVH
    }

    // Helper to safely get the standard BVH object, or nullptr if the layout is different
    tinybvh::BVH* get_standard_bvh() {
        // Only return a valid pointer if the active layout is the source/standard one
        if (active_bvh_view == source_bvh.get()) {
            return source_bvh.get();
        }
        return nullptr;
    }

    // A const version for read-only properties
    const tinybvh::BVH* get_standard_bvh() const {
        // Only return a valid pointer if the active layout is the source/standard one
        if (active_bvh_view == source_bvh.get()) {
            return source_bvh.get();
        }
        return nullptr;
    }

    // TODO: Layout converter method

    // Core builders (zero-copy)

    static std::unique_ptr<PyBVH> from_vertices(py::array_t<float, py::array::c_style> vertices_np,
        BuildQuality quality, float cost_traversal, float cost_intersection) {

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4 || vertices_np.shape(0) % 3 != 0) {
            throw std::runtime_error("Input vertices must be a 2D numpy array with shape (N*3, 4), where N is the number of triangles.");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto* bvh = wrapper->source_bvh.get();

        wrapper->source_geometry_refs.append(vertices_np);   // reference to the numpy array
        wrapper->quality = quality;

        bvh->c_trav = cost_traversal;
        bvh->c_int = cost_intersection;

        if (vertices_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
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

        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_indexed_mesh(
        py::array_t<float, py::array::c_style> vertices_np,
        py::array_t<uint32_t, py::array::c_style> indices_np,
        BuildQuality quality, float cost_traversal, float cost_intersection) {

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input vertices must be a 2D numpy array with shape (V, 4).");
        }
        if (indices_np.ndim() != 2 || indices_np.shape(1) != 3) {
            throw std::runtime_error("Input indices must be a 2D numpy array with shape (N, 3).");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto* bvh = wrapper->source_bvh.get();

        wrapper->source_geometry_refs.append(vertices_np);  // references to vertices numpy array
        wrapper->source_geometry_refs.append(indices_np);   // and indexes numpy array
        wrapper->quality = quality;

        bvh->c_trav = cost_traversal;
        bvh->c_int = cost_intersection;

        if (vertices_np.shape(0) == 0  || indices_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
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

        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_aabbs(py::array_t<float, py::array::c_style> aabbs_np,
        BuildQuality quality, float cost_traversal, float cost_intersection) {
        if (aabbs_np.ndim() != 3 || aabbs_np.shape(1) != 2 || aabbs_np.shape(2) != 3) {
            throw std::runtime_error("Input must be a 3D numpy array with shape (N, 2, 3).");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto* bvh = wrapper->source_bvh.get();

        wrapper->source_geometry_refs.append(aabbs_np);
        wrapper->quality = quality;
        wrapper->custom_type = CustomType::AABB;

        bvh->c_trav = cost_traversal;
        bvh->c_int = cost_intersection;

        // custom intersection functions
        bvh->customIntersect = aabb_intersect_callback;
        bvh->customIsOccluded = aabb_isoccluded_callback;

        if (aabbs_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
            return wrapper;
        }

        py::buffer_info aabbs_buf = aabbs_np.request();
        const auto* aabbs_ptr = static_cast<const float*>(aabbs_buf.ptr);
        const auto prim_count = static_cast<uint32_t>(aabbs_buf.shape[0]);

        // Pre-compute reciprocal extents for faster intersection tests
        auto inv_extents_np = py::array_t<float>(py::array::ShapeContainer({(py::ssize_t)prim_count, 3}));
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

        return wrapper;

        // `build_guard` goes out of scope, g_build_aabbs_ptr is set back to nullptr
    }

    // Convenience builders

    static std::unique_ptr<PyBVH> from_triangles(py::array_t<float, py::array::c_style | py::array::forcecast> tris_np,
        BuildQuality quality, float cost_traversal, float cost_intersection) {

        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }

        const size_t num_tris = tris_np.shape(0);

        auto wrapper = std::make_unique<PyBVH>();
        auto* bvh = wrapper->source_bvh.get();

        wrapper->quality = quality;

        bvh->c_trav = cost_traversal;
        bvh->c_int = cost_intersection;

        if (num_tris == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
            return wrapper;
        }

        //new numpy array to hold the reformatted data
        auto vertices_np = py::array_t<float>(py::array::ShapeContainer({(py::ssize_t)num_tris * 3, 4}));

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

        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_points(py::array_t<float, py::array::c_style | py::array::forcecast> points_np,
        float radius, BuildQuality quality, float cost_traversal, float cost_intersection) {

        if (points_np.ndim() != 2 || points_np.shape(1) != 3) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 3).");
        }
        if (radius <= 0.0f) {
            throw std::runtime_error("Point radius must be positive.");
        }

        auto wrapper = std::make_unique<PyBVH>();
        auto* bvh = wrapper->source_bvh.get();

        wrapper->quality = quality;
        wrapper->source_geometry_refs.append(points_np); // Store original points
        wrapper->custom_type = CustomType::Sphere;
        wrapper->sphere_radius = radius;

        bvh->customIntersect = sphere_intersect_callback;
        bvh->customIsOccluded = sphere_isoccluded_callback;
        bvh->c_trav = cost_traversal;
        bvh->c_int = cost_intersection;

        if (points_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
            return wrapper;
        }

        const size_t num_points = points_np.shape(0);
        const float* points_ptr = points_np.data();

        // Still need to create AABBs for the build process
        auto aabbs_np = py::array_t<float>(py::array::ShapeContainer({(py::ssize_t)num_points, 2, 3}));
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

        return wrapper;
    }

    // Interseciton methods

    float intersect(PyRay &py_ray) {

        if (!active_bvh_view || active_bvh_view->triCount == 0) {
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

        bool is_tlas = false;
        dispatch_bvh_call(active_bvh_view, [&](auto& bvh) {
            bvh.Intersect(ray);
            if constexpr (std::is_same_v<std::decay_t<decltype(bvh)>, tinybvh::BVH>) {
                is_tlas = bvh.isTLAS();
            }
        });

        if (ray.hit.t < py_ray.t) {
            py_ray.t = ray.hit.t; py_ray.u = ray.hit.u; py_ray.v = ray.hit.v;
#if INST_IDX_BITS == 32
            py_ray.prim_id = ray.hit.prim;
            py_ray.inst_id = is_tlas ? ray.hit.inst : (uint32_t)-1;
#else
            const uint32_t inst = ray.hit.prim >> INST_IDX_SHFT;
            py_ray.prim_id = ray.hit.prim & PRIM_IDX_MASK;
            py_ray.inst_id = is_tlas ? inst : (uint32_t)-1;
#endif
            return ray.hit.t;
        }
        return INFINITY;
    }

    py::array intersect_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> origins_np,
        py::array_t<float, py::array::c_style | py::array::forcecast> directions_np,
        py::object t_max_obj, py::object masks_obj)
    {
        // Shape checks
        if (origins_np.ndim() != 2 || origins_np.shape(1) != 3)
            throw std::runtime_error("Origins must be (N, 3) float32.");
        if (directions_np.ndim() != 2 || directions_np.shape(1) != 3)
            throw std::runtime_error("Directions must be (N, 3) float32.");
        if (origins_np.shape(0) != directions_np.shape(0))
            throw std::runtime_error("Origins and directions must have same N.");

        const py::ssize_t n_rays = origins_np.shape(0);
        const py::ssize_t stride = origins_np.shape(1);
        if (n_rays == 0)
            return py::array_t<HitRecord>(0);

        // Optional t_max / masks: keep arrays alive across GIL release
        const float* t_max_ptr = nullptr;  py::array_t<float>   t_max_np;
        if (!t_max_obj.is_none()) {
            t_max_np = py::cast<py::array_t<float>>(t_max_obj);
            if (t_max_np.ndim() != 1 || t_max_np.size() != n_rays)
                throw std::runtime_error("t_max must be a 1D float array of length N.");
            t_max_ptr = t_max_np.data();
        }

        const uint32_t* masks_ptr = nullptr;  py::array_t<uint32_t> masks_np;
        if (!masks_obj.is_none()) {
            masks_np = py::cast<py::array_t<uint32_t>>(masks_obj);
            if (masks_np.ndim() != 1 || masks_np.size() != n_rays)
                throw std::runtime_error("masks must be a 1D uint32 array of length N.");
            masks_ptr = masks_np.data();
        }

        // Custom geometry context: keep source arrays alive across GIL release
        py::array_t<float> aabbs_arr, inv_extents_arr, points_arr;
        CustomGeometryContext context{};
        if (custom_type == CustomType::AABB) {
            aabbs_arr = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            inv_extents_arr = py::cast<py::array_t<float>>(source_geometry_refs[1]);
            context.aabbs_ptr = aabbs_arr.data();
            context.inv_extents_ptr = inv_extents_arr.data();
        } else if (custom_type == CustomType::Sphere) {
            points_arr = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            context.points_ptr = points_arr.data();
            context.sphere_radius = sphere_radius;
        }

        // Early out: empty BVH
        if (!active_bvh_view || active_bvh_view->triCount == 0) {
            py::array_t<HitRecord> result(n_rays);
            auto* rp = result.mutable_data();
            for (py::ssize_t i = 0; i < n_rays; ++i)
                rp[i] = { (uint32_t)-1, (uint32_t)-1, INFINITY, 0.0f, 0.0f };
            return result;
        }

        const auto* origins_ptr = origins_np.data();
        const auto* directions_ptr = directions_np.data();

        std::vector<tinybvh::Ray> rays(n_rays);
        std::vector<HitRecord> out(n_rays);

        {
            py::gil_scoped_release release;

            // Initialize rays
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (py::ssize_t i = 0; i < n_rays; ++i) {
                const size_t iS = size_t(i) * size_t(stride);
                const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                const uint32_t mask_init = masks_ptr ? masks_ptr[i] : 0xFFFF;
                rays[i] = tinybvh::Ray(
                    {origins_ptr[iS+0], origins_ptr[iS+1], origins_ptr[iS+2]},
                    {directions_ptr[iS+0], directions_ptr[iS+1], directions_ptr[iS+2]},
                    t_init,
                    mask_init
                );
                if (custom_type != CustomType::None) {
                    rays[i].hit.auxData = &context;
                }
            }

            // Dispatch to active BVH
            dispatch_bvh_call(active_bvh_view, [&](auto& bvh) {
                using BVHType = std::decay_t<decltype(bvh)>;

                if constexpr (std::is_same_v<BVHType, tinybvh::BVH>) {
                    // Standard BVH: optionally use the 256-packet path when no custom geom
                    if (custom_type == CustomType::None) {
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; i += 256) {
                            const py::ssize_t end = std::min(i + 256, n_rays);
                            if (end - i == 256) {
                                #ifdef BVH_USEAVX
                                bvh.Intersect256RaysSSE(&rays[i]);
                                #else
                                bvh.Intersect256Rays(&rays[i]);
                                #endif
                            } else {
                                for (py::ssize_t j = i; j < end; ++j) bvh.Intersect(rays[j]);
                            }
                        }
                    } else {
                        // Standard BVH + custom geometry: scalar loop
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; ++i) bvh.Intersect(rays[i]);
                    }
                } else {
                    // Other layouts (SoA/AL/etc.): scalar loop
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(dynamic)
                    #endif
                    for (py::ssize_t i = 0; i < n_rays; ++i) bvh.Intersect(rays[i]);
                }
            });
        } // GIL re-acquired

        // Determine TLAS vs BLAS (only standard BVH exposes isTLAS())
        bool is_tlas = false;
        dispatch_bvh_call(active_bvh_view, [&](auto& bvh) {
            if constexpr (std::is_same_v<std::decay_t<decltype(bvh)>, tinybvh::BVH>)
                is_tlas = bvh.isTLAS();
        });

        // Fill POD output
        for (py::ssize_t i = 0; i < n_rays; ++i) {
            const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
            if (rays[i].hit.t < t_init) {
            #if INST_IDX_BITS == 32
                const uint32_t prim = rays[i].hit.prim;
                const uint32_t inst = is_tlas ? rays[i].hit.inst : (uint32_t)-1;
            #else
                const uint32_t prim = rays[i].hit.prim & PRIM_IDX_MASK;
                const uint32_t inst = is_tlas ? (rays[i].hit.prim >> INST_IDX_SHFT) : (uint32_t)-1;
            #endif
                out[i] = { prim, inst, rays[i].hit.t, rays[i].hit.u, rays[i].hit.v };
            } else {
                out[i] = { (uint32_t)-1, (uint32_t)-1, INFINITY, 0.0f, 0.0f };
            }
        }

        // Materialize numpy array and copy once
        py::array_t<HitRecord> result(n_rays);
        std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(HitRecord));
        return result;
    }

    bool is_occluded(const PyRay &py_ray) {

        if (!active_bvh_view || active_bvh_view->triCount == 0) {
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

        return dispatch_bvh_call(active_bvh_view, [&](auto& bvh) {
            return bvh.IsOccluded(ray);
        });
    }

    py::array_t<bool> is_occluded_batch(
        py::array_t<float, py::array::c_style> origins_np,
        py::array_t<float, py::array::c_style> directions_np,
        py::object t_max_obj, py::object masks_obj)
    {
        if (origins_np.ndim() != 2 || origins_np.shape(1) != 3)
            throw std::runtime_error("Origins must be (N, 3) float32.");
        if (directions_np.ndim() != 2 || directions_np.shape(1) != 3)
            throw std::runtime_error("Directions must be (N, 3) float32.");
        if (origins_np.shape(0) != directions_np.shape(0))
            throw std::runtime_error("Origins and directions must have same N.");

        const py::ssize_t n_rays = origins_np.shape(0);
        const py::ssize_t stride = origins_np.shape(1);

        py::array_t<bool> result(n_rays);
        if (n_rays == 0) return result;

        // Optional t_max / masks: keep arrays alive across GIL release
        const float* t_max_ptr = nullptr;  py::array_t<float>   t_max_np;
        if (!t_max_obj.is_none()) {
            t_max_np = py::cast<py::array_t<float>>(t_max_obj);
            if (t_max_np.ndim() != 1 || t_max_np.size() != n_rays)
                throw std::runtime_error("t_max must be a 1D float array of length N.");
            t_max_ptr = t_max_np.data();
        }

        const uint32_t* masks_ptr = nullptr;  py::array_t<uint32_t> masks_np;
        if (!masks_obj.is_none()) {
            masks_np = py::cast<py::array_t<uint32_t>>(masks_obj);
            if (masks_np.ndim() != 1 || masks_np.size() != n_rays)
                throw std::runtime_error("masks must be a 1D uint32 array of length N.");
            masks_ptr = masks_np.data();
        }

        // Custom geometry context: keep source arrays alive across GIL release
        py::array_t<float> aabbs_arr, inv_extents_arr, points_arr;
        CustomGeometryContext context{};
        if (custom_type == CustomType::AABB) {
            aabbs_arr       = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            inv_extents_arr = py::cast<py::array_t<float>>(source_geometry_refs[1]);
            context.aabbs_ptr       = aabbs_arr.data();
            context.inv_extents_ptr = inv_extents_arr.data();
        } else if (custom_type == CustomType::Sphere) {
            points_arr = py::cast<py::array_t<float>>(source_geometry_refs[0]);
            context.points_ptr   = points_arr.data();
            context.sphere_radius = sphere_radius;
        }

        auto* result_ptr = result.mutable_data();

        // Early out: empty BVH, all false
        if (!active_bvh_view || active_bvh_view->triCount == 0) {
            std::fill(result_ptr, result_ptr + n_rays, false);
            return result;
        }

        const auto* origins_ptr = origins_np.data();
        const auto* directions_ptr = directions_np.data();

        std::vector<uint8_t> occluded(n_rays, 0);
        {
            py::gil_scoped_release release;

            dispatch_bvh_call(active_bvh_view, [&](auto& bvh) {
                #ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (py::ssize_t i = 0; i < n_rays; ++i) {
                    const size_t iS = size_t(i) * size_t(stride);
                    const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                    const uint32_t mask_init = masks_ptr ? masks_ptr[i] : 0xFFFF;

                    tinybvh::Ray ray(
                        {origins_ptr[iS+0], origins_ptr[iS+1], origins_ptr[iS+2]},
                        {directions_ptr[iS+0], directions_ptr[iS+1], directions_ptr[iS+2]},
                        t_init,
                        mask_init
                    );
                    if (custom_type != CustomType::None)
                        ray.hit.auxData = &context;

                    occluded[i] = bvh.IsOccluded(ray) ? 1 : 0;
                }
            });
        } // GIL re-acquired

        for (py::ssize_t i = 0; i < n_rays; ++i)
            result_ptr[i] = (occluded[i] != 0);

        return result;
    }

    bool intersect_sphere(const py::object& center_obj, float radius) {

        if (!active_bvh_view || active_bvh_view->triCount == 0) {
            return false;
        }

        tinybvh::bvhvec3 center = py_obj_to_vec3(center_obj);

        bool supported = false;
        bool result = dispatch_bvh_call(active_bvh_view, [&](auto& bvh) {
            using BvhType = std::decay_t<decltype(bvh)>;
            if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                supported = true;
                return bvh.IntersectSphere(center, radius);
            }
            return false; // Dummy return
        });

        if (!supported) {
            throw std::runtime_error("Sphere intersection is not supported for the current BVH layout.");
        }
        return result;
    }

    // Refitting, TLAS

    void refit() {
        // the standard BVH is the only type that can be refit
        tinybvh::BVH* bvh = get_standard_bvh();
        if (!bvh) {
            throw std::runtime_error("Refit is only supported for the standard BVH layout.");
        }

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
            return std::make_unique<PyBVH>(); // Return an empty BVH
        }

        // Create a new PyBVH wrapper. The constructor correctly initializes active_bvh_view
        // with a new standard tinybvh::BVH, which is required for a TLAS
        auto wrapper = std::make_unique<PyBVH>();

        // Resize the member vectors that will hold the data for the build
        wrapper->instances_data.resize(inst_count);
        wrapper->blas_pointers.reserve(blases_py.size());

        // Extract the BVHBase* pointers from the list of PyBVH objects
        // This gets the pointer from any layout (BVH, SoA, etc.)
        for (const auto& blas_obj : blases_py) {
            auto& py_blas = blas_obj.cast<PyBVH&>();
            wrapper->blas_pointers.push_back(py_blas.active_bvh_view);
        }

        // Populate the instance data vector
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

        // Get the typed pointer to the BVH and call the Build function
        // We must cast the BVHBase* to the specific tinybvh::BVH* that has the TLAS build method
        auto* tlas_bvh = wrapper->source_bvh.get(); // A TLAS is always a standard BVH
        if (!tlas_bvh) {
            // This should not happen since the constructor creates a standard BVH
            throw std::logic_error("Internal error: TLAS build requires a standard BVH layout.");
        }

        tlas_bvh->Build(
            wrapper->instances_data.data(),
            static_cast<uint32_t>(inst_count),
            wrapper->blas_pointers.data(),
            static_cast<uint32_t>(wrapper->blas_pointers.size())
        );

        return wrapper;
    }
    // Load / save

    static std::unique_ptr<PyBVH> load(const py::object& filepath_obj, py::array_t<float,
        py::array::c_style | py::array::forcecast> vertices_np, py::object indices_obj) { // py::object to accept an array or None

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Vertex data for loading must be a 2D numpy array with shape (V, 4).");
        }

        auto filepath = py::str(filepath_obj).cast<std::string>();
        auto wrapper = std::make_unique<PyBVH>();

        // Create a standard BVH to load into
        auto* bvh = wrapper->source_bvh.get();
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

        // Move the loaded BVH into the base pointer
        return wrapper;

    }

    void save(const py::object& filepath_obj) {
        if (!active_bvh_view || active_bvh_view->layout != tinybvh::BVHBase::LAYOUT_BVH) {
            throw std::runtime_error("Saving is only supported for the standard BVH layout.");
        }
        auto* bvh = static_cast<tinybvh::BVH*>(active_bvh_view);

        auto filepath = py::str(filepath_obj).cast<std::string>();
        bvh->Save(filepath.c_str());
    }

    // Advanced manipulation methods

    void optimize(unsigned int iterations, bool extreme, bool stochastic) {
        tinybvh::BVH* bvh = get_standard_bvh();
        if (!bvh) {
            throw std::runtime_error("Optimization is only supported for the standard BVH layout.");
        }

        // TODO: Why does this produce garbage SOMETIMES ? (Kept your original comment)

        tinybvh::BVH_Verbose verbose_bvh;

        // The verbose BVH needs to be allocated with enough space for splitting
        verbose_bvh.bvhNode = (tinybvh::BVH_Verbose::BVHNode*)verbose_bvh.AlignedAlloc(
            sizeof(tinybvh::BVH_Verbose::BVHNode) * bvh->triCount * 3);
        verbose_bvh.allocatedNodes = bvh->triCount * 3;
        verbose_bvh.ConvertFrom(*bvh);

        // Optimize tree structure
        verbose_bvh.Optimize(iterations, extreme, stochastic);

        // Convert dense BVH_Verbose back to standard BVH format
        bvh->ConvertFrom(verbose_bvh, true /* compact */);

        // update quality state of the Python wrapper
        quality = bvh->refittable ? BuildQuality::Balanced : BuildQuality::High;
    }

    void set_opacity_maps(py::array_t<uint32_t, py::array::c_style> map_data, uint32_t N) {
        if (map_data.ndim() != 1) {
            throw std::runtime_error("Opacity map data must be a 1D uint32 numpy array.");
        }
        // tinybvh expects a specific size: N*N bits per triangle
        // The array should contain (triCount * N * N) / 32 uint32_t values
        const size_t expected_size = (active_bvh_view->triCount * N * N + 31) / 32;
        if (static_cast<size_t>(map_data.size()) != expected_size) {
            throw std::runtime_error("Opacity map data has incorrect size for the given N and primitive count.");
        }

        opacity_map_ref = map_data; // Keep reference
        active_bvh_view->SetOpacityMicroMaps(opacity_map_ref.mutable_data(), N);
    }

    void compact() {
        tinybvh::BVH* bvh = get_standard_bvh();
        if (!bvh) {
            throw std::runtime_error("compact() is only supported for the standard BVH layout.");
        }
        bvh->Compact();
    }

    void split_leaves(uint32_t max_prims) {
        tinybvh::BVH* bvh = get_standard_bvh();
        if (!bvh) {
            throw std::runtime_error("split_leaves() is only supported for the standard BVH layout.");
        }
        if (bvh->triCount == 0) throw std::runtime_error("BVH is not initialized.");
        if (bvh->usedNodes + bvh->triCount > bvh->allocatedNodes) {
            throw std::runtime_error("Cannot split leaves: not enough allocated node capacity.");
        }
        bvh->SplitLeafs(max_prims);
    }

    void combine_leaves() {
        tinybvh::BVH* bvh = get_standard_bvh();
        if (!bvh) {
            throw std::runtime_error("combine_leaves() is only supported for the standard BVH layout.");
        }
        if (bvh->triCount == 0) throw std::runtime_error("BVH is not initialized.");
        bvh->CombineLeafs();
    }

};



// =============================================== pybind11 module =====================================================


PYBIND11_MODULE(pytinybvh, m) {
    m.doc() = "Python bindings for the tinybvh library";

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

    // Build quality and Geometry type structs
    py::enum_<BuildQuality>(m, "BuildQuality", "Enum for selecting BVH build quality.")
        .value("Quick", BuildQuality::Quick, "Fastest build, lower quality queries.")
        .value("Balanced", BuildQuality::Balanced, "Balanced build time and query performance (default).")
        .value("High", BuildQuality::High, "Slowest build (uses spatial splits), highest quality queries.");

    py::enum_<GeometryType>(m, "GeometryType")
        .value("Triangles", GeometryType::Triangles, "The BVH was built over a triangle mesh.")
        .value("AABBs", GeometryType::AABBs, "The BVH was built over custom Axis-Aligned Bounding Boxes.")
        .value("Spheres", GeometryType::Spheres, "The BVH was built over a point cloud with a radius (spheres).");

    py::enum_<tinybvh::BVHBase::BVHType> layout_enum(m, "Layout",
        "Enum for the internal memory layout of the BVH.");

        layout_enum.value("Standard", tinybvh::BVHBase::LAYOUT_BVH,
            "Standard BVH layout (default).");
        layout_enum.value("SoA", tinybvh::BVHBase::LAYOUT_BVH_SOA,
            "Structure of Arrays layout, optimized for SSE/NEON traversal.");
        layout_enum.export_values();

        // TODO: Add other layouts

    // Main Python classes
    py::class_<PyRay>(m, "Ray", "Represents a ray for intersection queries.")

        .def(py::init([](const py::object& origin_obj, const py::object& direction_obj, float t, uint32_t mask) {
             return PyRay{
                py_obj_to_vec3(origin_obj),
                py_obj_to_vec3(direction_obj),
                t, 0.f, 0.f, (uint32_t)-1, (uint32_t)-1, mask
             };
        }), "origin"_a, "direction"_a, "t"_a = 1e30f, "mask"_a = 0xFFFF)

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

            if (r.prim_id != (uint32_t)-1) {
                return "<pytinybvh.Ray (origin=" + origin_s + " dir=" + dir_s +
                       ", Hit inst " + std::to_string(r.inst_id) +
                       " prim " + std::to_string(r.prim_id) +
                       " at t=" + std::to_string(r.t) + ")>";
            }
            return "<pytinybvh.Ray (origin=" + origin_s + " dir=" + dir_s + ", Miss)>";
        });


    py::class_<PyBVH>(m, "BVH", "A Bounding Volume Hierarchy for fast ray intersections.")

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
            py::arg("cost_traversal") = C_TRAV, py::arg("cost_intersection") = C_INT)

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
            py::arg("cost_traversal") = C_TRAV, py::arg("cost_intersection") = C_INT)

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
            py::arg("cost_traversal") = C_TRAV, py::arg("cost_intersection") = C_INT)

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
            py::arg("cost_traversal") = C_TRAV, py::arg("cost_intersection") = C_INT)

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
            py::arg("cost_traversal") = C_TRAV, py::arg("cost_intersection") = C_INT)

        // TODO: Layout converter method

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

                Returns:
                    numpy.ndarray: A structured array of shape (N,) with dtype
                        [('prim_id', '<u4'), ('inst_id', '<u4'), ('t', '<f4'), ('u', '<f4'), ('v', '<f4')].
                        For misses, prim_id and inst_id are -1 and t is infinity.
                        For TLAS hits, inst_id is the instance index and prim_id is the primitive
                        index within that instance's BLAS.
           ))",
           py::arg("origins"), py::arg("directions"), py::arg("t_max") = py::none(), py::arg("masks") = py::none())

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

                Returns:
                    numpy.ndarray: A boolean array of shape (N,) where `True` indicates occlusion.
           ))",
           py::arg("origins"), py::arg("directions"), py::arg("t_max") = py::none(), py::arg("masks") = py::none())

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

        // Getters for the main data

        .def_property_readonly("nodes", [](const PyBVH &self) -> py::array {
            const tinybvh::BVH* bvh = self.get_standard_bvh(); // using the const helper
            if (!bvh || bvh->usedNodes == 0) {
                py::dtype dt = py::dtype::of<tinybvh::BVH::BVHNode>();
                return py::array(dt, {0}, {}); // Return empty array with correct dtype
            }
            // Expose the view as before, but using the checked pointer 'bvh'
            return py::array_t<tinybvh::BVH::BVHNode>(
                bvh->usedNodes, bvh->bvhNode, py::cast(self)
            );
            }, "The structured numpy array of BVH nodes (only available for standard layout).")

        .def_property_readonly("prim_indices", [](const PyBVH &self) -> py::array {
            const tinybvh::BVH* bvh = self.get_standard_bvh(); // using the const helper
            if (!bvh || bvh->idxCount == 0) {
                return py::array_t<uint32_t>(0); // Return empty uint32 array
            }
            return py::array_t<uint32_t>(
                bvh->idxCount,
                bvh->primIdx,
                py::cast(self)
            );
        }, "The array of primitive indices (only available for standard layout).")

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
             "iterations"_a = 25, "extreme"_a = false, "stochastic"_a = false,
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
             ))")

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

        // Read-write properties

        .def_property("cost_traversal",
            [](const PyBVH &self) {
                if (!self.active_bvh_view) throw std::runtime_error("BVH is not initialized.");
                return self.active_bvh_view->c_trav;
            },
            [](PyBVH &self, float value) {
                if (!self.active_bvh_view) throw std::runtime_error("BVH is not initialized.");
                if (value < 0) throw std::runtime_error("Traversal cost must be non-negative.");
                self.active_bvh_view->c_trav = value;
            },
            "The traversal cost used in the Surface Area Heuristic (SAH) calculation during the build.")

        .def_property("cost_intersection",
            [](const PyBVH &self) {
                if (!self.active_bvh_view) throw std::runtime_error("BVH is not initialized.");
                return self.active_bvh_view->c_int;
            },
            [](PyBVH &self, float value) {
                if (!self.active_bvh_view) throw std::runtime_error("BVH is not initialized.");
                if (value < 0) throw std::runtime_error("Intersection cost must be non-negative.");
                self.active_bvh_view->c_int = value;
            },
            "The intersection cost used in the Surface Area Heuristic (SAH) calculation during the build.")

        // Read-only properties

        .def_property_readonly("node_count", [](const PyBVH &self){ return self.active_bvh_view->usedNodes; }, "Total number of nodes in the BVH.")

        .def_property_readonly("prim_count", [](const PyBVH &self){ return self.active_bvh_view->triCount; }, "Total number of primitives in the BVH.")

        .def_property_readonly("aabb_min", [](PyBVH &self) -> py::array {
            if (!self.active_bvh_view) return py::array_t<float>(0);
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMin data
            std::vector<py::ssize_t> shape = {3};
            return py::array_t<float>(
                shape,
                &self.active_bvh_view->aabbMin.x,   // data pointer
                py::cast(self)          // owner
            );
        }, "The minimum corner of the root axis-aligned bounding box.")

        .def_property_readonly("aabb_max", [](PyBVH &self) -> py::array {
            if (!self.active_bvh_view) return py::array_t<float>(0);
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMax data
            std::vector<py::ssize_t> shape = {3};
            return py::array_t<float>(
                shape,
                &self.active_bvh_view->aabbMax.x,   // data pointer
                py::cast(self)          // owner
            );
        }, "The maximum corner of the root axis-aligned bounding box.")

        .def_property_readonly("quality", [](const PyBVH &self) { return self.quality; },
           "The build quality level used to construct the BVH.")

        .def_property_readonly("leaf_count", [](const PyBVH &self) {
            if (!self.active_bvh_view) return 0;
            return dispatch_bvh_call(self.active_bvh_view, [](auto& bvh) {
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
            }, "The total number of leaf nodes in the BVH (not supported by all layouts).")

        .def_property_readonly("sah_cost", [](const PyBVH &self) {
            if (!self.active_bvh_view || self.active_bvh_view->triCount == 0) {
                return INFINITY;
            }
            // Dispatch the call to the appropriate concrete BVH type
            // Calling with SAHCost(0) is compatible with all layouts that have the method
            return dispatch_bvh_call(self.active_bvh_view, [](auto& bvh) {
                return bvh.SAHCost(0);
            });
            }, R"((Calculates the Surface Area Heuristic (SAH) cost of the BVH. Lower is better.))")

        .def_property_readonly("epo_cost", [](const PyBVH &self) {
            if (!self.active_bvh_view || self.active_bvh_view->triCount == 0) return 0.0f;
            return dispatch_bvh_call(self.active_bvh_view, [](auto& bvh) {
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.EPOCost();
                } else {
                    return 0.0f; // EPO is not defined for other layouts
                }
            });
            }, "Calculates the EPO cost of the BVH (only supported for standard layout).")

        .def_property_readonly("layout", [](const PyBVH &self) {
            return self.active_bvh_view->layout;
        }, "The current memory layout of the BVH.")

        .def_property_readonly("is_tlas", [](const PyBVH &self) {
            if (!self.active_bvh_view) return false;
            return dispatch_bvh_call(self.active_bvh_view, [](auto& bvh){
                using BvhType = std::decay_t<decltype(bvh)>;
                // Only the standard BVH layout can be a TLAS
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.isTLAS();
                }
                // All other layouts are for geometry, so they are BLASes
                return false;
            });
            }, "Returns True if the BVH is a Top-Level Acceleration Structure (TLAS).")

        .def_property_readonly("is_blas", [](const PyBVH &self) {
            if (!self.active_bvh_view) return true; // An uninitialized/empty BVH can be considered a BLAS
            // The logic is simply the inverse of is_tlas.
            return dispatch_bvh_call(self.active_bvh_view, [](auto& bvh){
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.isBLAS();
                }
                return true; // All other layouts are BLASes
            });
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
            return self.active_bvh_view ? !self.active_bvh_view->may_have_holes : true;
        }, "Returns True if the BVH node and index arrays are contiguous in memory.")

        .def("__repr__", [](const PyBVH& self) {
            if (!self.active_bvh_view) return std::string("<pytinybvh.BVH (uninitialized)>");

            std::string repr = "<pytinybvh.BVH (";
            repr += std::to_string(self.active_bvh_view->triCount) + " primitives, ";
            repr += std::to_string(self.active_bvh_view->usedNodes) + " nodes, ";

            // Geometry Type
            switch (self.custom_type) {
                case PyBVH::CustomType::AABB:   repr += "Geometry: AABBs, "; break;
                case PyBVH::CustomType::Sphere: repr += "Geometry: Spheres, "; break;
                default: break;
            }

            // TODO: make it a property insyead of replicating the logic from the 'is_tlas' property
            bool tlas_check = dispatch_bvh_call(self.active_bvh_view, [](auto& bvh){
                using BvhType = std::decay_t<decltype(bvh)>;
                if constexpr (std::is_same_v<BvhType, tinybvh::BVH>) {
                    return bvh.isTLAS();
                }
                return false; // Other layouts are always BLASes
            });

            if (tlas_check) {
                repr += "Type: TLAS, ";
            } else {
                repr += "Type: BLAS, ";
                std::string quality_str;
                switch(self.quality) {
                    case BuildQuality::Quick: quality_str = "Quick"; break;
                    case BuildQuality::Balanced: quality_str = "Balanced"; break;
                    case BuildQuality::High: quality_str = "High"; break;
                }
                repr += "Quality: " + quality_str + ", ";
            }

            // Layout and Compactness
            repr += "Layout: " + layout_to_string(self.active_bvh_view->layout) + ", ";
            repr += std::string("Status: ") + (self.active_bvh_view->may_have_holes ? "Not compact" : "Compact");

            repr += ")>";
            return repr;
        })

    ;
}