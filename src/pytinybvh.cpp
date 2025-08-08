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


// Helper to extract 3 floats from a Python object (list, tuple, numpy array)
tinybvh::bvhvec3 py_obj_to_vec3(const py::object& obj) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = py::cast<py::sequence>(obj);
        if (seq.size() != 3) {
            throw std::runtime_error("Input sequence must have 3 elements for a 3D vector.");
        }
        return tinybvh::bvhvec3(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>());
    }
    if (py::isinstance<py::array>(obj)) {
        py::array_t<float> arr = py::cast<py::array_t<float>>(obj);
        if (arr.ndim() != 1 || arr.size() != 3) {
            throw std::runtime_error("Input numpy array must have shape (3,) for a 3D vector.");
        }
        return tinybvh::bvhvec3(arr.at(0), arr.at(1), arr.at(2));
    }
    throw std::runtime_error("Input must be a list, tuple, or numpy array of 3 floats.");
}


// thread_local pointer to pass the AABB data to the C-style callbacks
thread_local const float* g_aabbs_ptr = nullptr;
thread_local const float* g_points_ptr = nullptr;
thread_local float g_sphere_radius = 0.0f;
thread_local const float* g_inv_extents_ptr = nullptr;

// C-style callback for building from AABBs
void aabb_build_callback(const unsigned int i, tinybvh::bvhvec3& bmin, tinybvh::bvhvec3& bmax) {
    if (!g_aabbs_ptr) throw std::runtime_error("Internal error: AABB data pointer is null in build callback.");
    const size_t offset = i * 6; // 2 vectors * 3 floats
    bmin.x = g_aabbs_ptr[offset + 0]; bmin.y = g_aabbs_ptr[offset + 1]; bmin.z = g_aabbs_ptr[offset + 2];
    bmax.x = g_aabbs_ptr[offset + 3]; bmax.y = g_aabbs_ptr[offset + 4]; bmax.z = g_aabbs_ptr[offset + 5];
}

// C-style callback for intersecting AABBs
bool aabb_intersect_callback(tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!g_aabbs_ptr) throw std::runtime_error("Internal error: AABB data pointer is null in intersect callback.");

    const size_t offset = prim_id * 6;
    const tinybvh::bvhvec3 bmin = {g_aabbs_ptr[offset], g_aabbs_ptr[offset + 1], g_aabbs_ptr[offset + 2]};
    const tinybvh::bvhvec3 bmax = {g_aabbs_ptr[offset + 3], g_aabbs_ptr[offset + 4], g_aabbs_ptr[offset + 5]};

    float t = tinybvh::tinybvh_intersect_aabb(ray, bmin, bmax);

    if (t < ray.hit.t) {
        ray.hit.t = t;
        ray.hit.prim = prim_id;

        // Calculate UV coordinates on the hit face
        const tinybvh::bvhvec3 hit_point = ray.O + ray.D * t;

        // Use pre-computed reciprocal extent
        const size_t inv_offset = prim_id * 3;
        const tinybvh::bvhvec3 inv_extent = {g_inv_extents_ptr[inv_offset], g_inv_extents_ptr[inv_offset + 1], g_inv_extents_ptr[inv_offset + 2]};

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
        }

        return true;
    }
    return false;
}

// C-style callback for occlusion testing with AABBs
bool aabb_isoccluded_callback(const tinybvh::Ray& ray, const unsigned int prim_id) {
    if (!g_aabbs_ptr) throw std::runtime_error("Internal error: AABB data pointer is null in occlusion callback.");

    const size_t offset = prim_id * 6;
    const tinybvh::bvhvec3 bmin = {g_aabbs_ptr[offset], g_aabbs_ptr[offset + 1], g_aabbs_ptr[offset + 2]};
    const tinybvh::bvhvec3 bmax = {g_aabbs_ptr[offset + 3], g_aabbs_ptr[offset + 4], g_aabbs_ptr[offset + 5]};

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
    if (!g_points_ptr) throw std::runtime_error("Internal error: Sphere data pointer is null.");

    // Pre-computed reciprocals
    constexpr float PI = 3.1415926535f;
    constexpr float INV_PI = 1.0f / PI;
    constexpr float INV_TWO_PI = 1.0f / (2.0f * PI);

    // Get sphere center
    const size_t offset = prim_id * 3;
    const tinybvh::bvhvec3 center = {g_points_ptr[offset], g_points_ptr[offset + 1], g_points_ptr[offset + 2]};

    // Ray-sphere intersection
    const tinybvh::bvhvec3 oc = ray.O - center;
    const float a = tinybvh_dot(ray.D, ray.D);
    const float b = 2.0f * tinybvh_dot(oc, ray.D);
    const float c = tinybvh_dot(oc, oc) - g_sphere_radius * g_sphere_radius;
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
    if (!g_points_ptr) throw std::runtime_error("Internal error: Sphere data pointer is null.");

    const size_t offset = prim_id * 3;
    const tinybvh::bvhvec3 center = {g_points_ptr[offset], g_points_ptr[offset + 1], g_points_ptr[offset + 2]};

    const tinybvh::bvhvec3 oc = ray.O - center;
    const float a = tinybvh::tinybvh_dot(ray.D, ray.D);
    const float b = 2.0f * tinybvh::tinybvh_dot(oc, ray.D);
    const float c = tinybvh::tinybvh_dot(oc, oc) - g_sphere_radius * g_sphere_radius;
    const float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    const float t = (-b - sqrt(discriminant)) * 0.5f; // Since a=1.0
    return (t > 1e-6f && t < ray.hit.t);
}



// Enum for build quality selection
enum class BuildQuality {
    Quick,
    Balanced,
    High
};

// C++ class to wrap the tinybvh::BVH object and its associated data: this is the object held by the Python BVH class
struct PyBVH {
    std::unique_ptr<tinybvh::BVH> bvh;

    py::list source_geometry_refs; // refs to the numpy arrays that contains the source data (vertices, indices)
    // this prevents Python's garbage collector from freeing the memory while the C++ BVH object might still be using it

    std::vector<tinybvh::BLASInstance> instances_data;  // ref to keep BLAS instances
    std::vector<tinybvh::BVHBase*> blas_pointers;

    BuildQuality quality = BuildQuality::Balanced;

    enum class CustomType { None, AABB, Sphere };
    CustomType custom_type = CustomType::None;
    float sphere_radius = 0.0f;

    PyBVH() : bvh(std::make_unique<tinybvh::BVH>()) {}

    // Core builders (zero-copy)

    static std::unique_ptr<PyBVH> from_vertices(py::array_t<float, py::array::c_style> vertices_np, BuildQuality quality) {
        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4 || vertices_np.shape(0) % 3 != 0) {
            throw std::runtime_error("Input vertices must be a 2D numpy array with shape (N*3, 4), where N is the number of triangles.");
        }

        auto wrapper = std::make_unique<PyBVH>();
        wrapper->source_geometry_refs.append(vertices_np);   // reference to the numpy array
        wrapper->quality = quality;

        if (vertices_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
            return wrapper;
        }

        py::buffer_info vertices_buf = vertices_np.request();
        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
        const uint32_t prim_count = static_cast<uint32_t>(vertices_buf.shape[0] / 3);

        switch (quality) {
            case BuildQuality::Quick:
                wrapper->bvh->BuildQuick(vertices_ptr, prim_count);
                break;
            case BuildQuality::High:
                wrapper->bvh->BuildHQ(vertices_ptr, prim_count);
                break;
            case BuildQuality::Balanced:
            default:
                wrapper->bvh->Build(vertices_ptr, prim_count);
                break;
        }

        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_indexed_mesh(
        py::array_t<float, py::array::c_style> vertices_np,
        py::array_t<uint32_t, py::array::c_style> indices_np,
        BuildQuality quality) {

        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input vertices must be a 2D numpy array with shape (V, 4).");
        }
        if (indices_np.ndim() != 2 || indices_np.shape(1) != 3) {
            throw std::runtime_error("Input indices must be a 2D numpy array with shape (N, 3).");
        }

        auto wrapper = std::make_unique<PyBVH>();
        wrapper->source_geometry_refs.append(vertices_np);  // references to vertices numpy array
        wrapper->source_geometry_refs.append(indices_np);   // and indexes numpy array
        wrapper->quality = quality;

        if (vertices_np.shape(0) == 0  || indices_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
            return wrapper;
        }

        py::buffer_info vertices_buf = vertices_np.request();
        py::buffer_info indices_buf = indices_np.request();

        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
        auto indices_ptr = static_cast<const uint32_t*>(indices_buf.ptr);
        const uint32_t prim_count = static_cast<uint32_t>(indices_buf.shape[0]);

        switch (quality) {
            case BuildQuality::Quick:
                // tinybvh doesn't have an indexed BuildQuick so fall back to Balanced
                wrapper->bvh->Build(vertices_ptr, indices_ptr, prim_count);
                break;
            case BuildQuality::High:
                wrapper->bvh->BuildHQ(vertices_ptr, indices_ptr, prim_count);
                break;
            case BuildQuality::Balanced:
            default:
                wrapper->bvh->Build(vertices_ptr, indices_ptr, prim_count);
                break;
        }

        return wrapper;
    }

    // Convenience builders (with copying)

    static std::unique_ptr<PyBVH> from_triangles(py::array_t<float, py::array::c_style | py::array::forcecast> tris_np, BuildQuality quality) {

        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }

        const size_t num_tris = tris_np.shape(0);

        // this must be done inside this function so vertices_np stays alive
        auto wrapper = std::make_unique<PyBVH>();
        wrapper->quality = quality;

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
                wrapper->bvh->BuildQuick(vertices_ptr, static_cast<uint32_t>(num_tris));
                break;
            case BuildQuality::High:
                wrapper->bvh->BuildHQ(vertices_ptr, static_cast<uint32_t>(num_tris));
                break;
            case BuildQuality::Balanced:
            default:
                wrapper->bvh->Build(vertices_ptr, static_cast<uint32_t>(num_tris));
                break;
        }

        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_points(py::array_t<float, py::array::c_style | py::array::forcecast> points_np, float radius, BuildQuality quality) {
        if (points_np.ndim() != 2 || points_np.shape(1) != 3) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 3).");
        }
        if (radius <= 0.0f) {
            throw std::runtime_error("Point radius must be positive.");
        }

        auto wrapper = std::make_unique<PyBVH>();
        wrapper->quality = quality;
        wrapper->source_geometry_refs.append(points_np); // Store original points
        wrapper->custom_type = CustomType::Sphere;
        wrapper->sphere_radius = radius;
        wrapper->bvh->customIntersect = sphere_intersect_callback;
        wrapper->bvh->customIsOccluded = sphere_isoccluded_callback;

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

        g_aabbs_ptr = aabbs_ptr; // use the AABB ptr for the build
        try {
            wrapper->bvh->Build(aabb_build_callback, static_cast<uint32_t>(num_points));
        } catch (...) {
            g_aabbs_ptr = nullptr;
            throw;
        }
        g_aabbs_ptr = nullptr;

        return wrapper;
    }

    static std::unique_ptr<PyBVH> from_aabbs(py::array_t<float, py::array::c_style | py::array::forcecast> aabbs_np, BuildQuality quality) {
        if (aabbs_np.ndim() != 3 || aabbs_np.shape(1) != 2 || aabbs_np.shape(2) != 3) {
            throw std::runtime_error("Input must be a 3D numpy array with shape (N, 2, 3).");
        }

        auto wrapper = std::make_unique<PyBVH>();
        wrapper->source_geometry_refs.append(aabbs_np);
        wrapper->quality = quality;
        wrapper->custom_type = CustomType::AABB;

        // custom intersection functions
        wrapper->bvh->customIntersect = aabb_intersect_callback;
        wrapper->bvh->customIsOccluded = aabb_isoccluded_callback;

        if (aabbs_np.shape(0) == 0) {
            // default wrapper is an empty BVH
            // bvh->triCount is 0, bvh->usedNodes is 0.
            return wrapper;
        }

        py::buffer_info aabbs_buf = aabbs_np.request();
        const auto* aabbs_ptr = static_cast<const float*>(aabbs_buf.ptr);
        const uint32_t prim_count = static_cast<uint32_t>(aabbs_buf.shape[0]);

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

        // Use the original AABBs for the build callback
        g_aabbs_ptr = aabbs_ptr;

        try {
            // BuildQuick and BuildHQ don't support custom AABB getters, so default to Balanced
            switch (quality) {
                case BuildQuality::High:
                case BuildQuality::Quick:
                case BuildQuality::Balanced:
                default:
                     // Pass the C-style function pointer, not the lambda
                     wrapper->bvh->Build(aabb_build_callback, prim_count);
                     break;
            }
        } catch (...) {
            // ensure global pointer is cleared even if an exception occurs
            g_aabbs_ptr = nullptr;
            throw;
        }

        // clear the pointer after the build is complete
        g_aabbs_ptr = nullptr;

        return wrapper;
    }

    // Other methods

    void refit() {
        if (!bvh || !bvh->refittable) {
            throw std::runtime_error("BVH is not refittable (it might have spatial splits, i.e. built with BuildQuality.High).");
        }
        // Refit needs original vertex data
        // `bvh->verts` slice already points to it
        bvh->Refit();
    }

    // TODO: optimise()
};

// pybind11 wrapper for tinybvh::Ray to be used in intersection queries
struct PyRay {
    tinybvh::bvhvec3 origin;
    tinybvh::bvhvec3 direction;
    float t = 1e30f;
    float u = 0.0f, v = 0.0f;
    uint32_t prim_id = -1;
    uint32_t inst_id = -1;
    uint32_t mask = 0xFFFF;
};


// =====================================================================================================================


PYBIND11_MODULE(pytinybvh, m) {
    m.doc() = "Python bindings for the tinybvh library";

    // HitRecord struct and its numpy dtype
    struct HitRecord {
        uint32_t prim_id;
        uint32_t inst_id;
        float t, u, v;
    };
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

    py::enum_<BuildQuality>(m, "BuildQuality", "Enum for selecting BVH build quality.")
        .value("Quick", BuildQuality::Quick, "Fastest build, lower quality queries.")
        .value("Balanced", BuildQuality::Balanced, "Balanced build time and query performance (default).")
        .value("High", BuildQuality::High, "Slowest build (uses spatial splits), highest quality queries.");

    py::class_<tinybvh::bvhvec3>(m, "vec3", "A 3D vector with float components.")
        .def(py::init<float, float, float>(), "x"_a=0.f, "y"_a=0.f, "z"_a=0.f)
        .def_readwrite("x", &tinybvh::bvhvec3::x, "X-component of the vector.")
        .def_readwrite("y", &tinybvh::bvhvec3::y, "Y-component of the vector.")
        .def_readwrite("z", &tinybvh::bvhvec3::z, "Z-component of the vector.")
        .def("__repr__", [](const tinybvh::bvhvec3 &v) {
            return "vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

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
            if (r.prim_id != (uint32_t)-1) {
                return "<pytinybvh.Ray hit primitive " + std::to_string(r.prim_id) + " at t=" + std::to_string(r.t) + ">";
            }
            return std::string("<pytinybvh.Ray miss>");
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
            py::arg("vertices").noconvert(), py::arg("quality") = BuildQuality::Balanced)

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
            py::arg("vertices").noconvert(),
            py::arg("indices").noconvert(),
            py::arg("quality") = BuildQuality::Balanced)

        // Convenience Builders
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
            py::arg("triangles"),
            py::arg("quality") = BuildQuality::Balanced)

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
            py::arg("points"),
            py::arg("radius") = 1e-5f,
            py::arg("quality") = BuildQuality::Balanced)

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
            py::arg("aabbs"),
            py::arg("quality") = BuildQuality::Balanced)

        .def_static("build_tlas", [](
            py::array instances_np, // generic array, we check dtype manually
            const py::list& blases_py
        ) {
            // validate
            if (instances_np.ndim() != 1) {
                throw std::runtime_error("Instances must be a 1D structured numpy array.");
            }
            // bit verbose but it's fine
            py::dict fields = instances_np.dtype().attr("fields").cast<py::dict>();
            if (!fields.contains("transform") || !fields.contains("blas_id")) {
                 throw std::runtime_error("Instance dtype must contain 'transform' and 'blas_id'.");
            }

            const py::ssize_t inst_count = instances_np.shape(0);
            if (inst_count == 0) {
                // Handle empty scene
                return std::make_unique<PyBVH>();
            }

            auto wrapper = std::make_unique<PyBVH>();
            wrapper->instances_data.resize(inst_count);

            // Extract BLAS pointers
            wrapper->blas_pointers.reserve(blases_py.size());
            for (const auto& blas_obj : blases_py) {
                wrapper->blas_pointers.push_back(blas_obj.cast<PyBVH&>().bvh.get());
            }

            for (py::ssize_t i = 0; i < inst_count; ++i) {

                auto record = instances_np[py::int_(i)];
                auto transform_arr = record["transform"].cast<py::array_t<float>>();
                wrapper->instances_data[i].blasIdx = record["blas_id"].cast<uint32_t>();

                // Optional support for mask
                if (fields.contains("mask")) {
                    wrapper->instances_data[i].mask = record["mask"].cast<uint32_t>();
                }

                // Copy the transform matrix (it's fine we only do this once)
                std::memcpy(wrapper->instances_data[i].transform.cell, transform_arr.data(), 16 * sizeof(float));
            }

            wrapper->source_geometry_refs.append(instances_np); // keep instance data alive
            wrapper->source_geometry_refs.append(blases_py);    // keep BLASes alive

            wrapper->bvh->Build(
                wrapper->instances_data.data(),
                static_cast<uint32_t>(inst_count),
                wrapper->blas_pointers.data(),
                static_cast<uint32_t>(wrapper->blas_pointers.size())
            );

            return wrapper;

        }, py::arg("instances").noconvert(), py::arg("BLASes"),
           R"((
                Builds a Top-Level Acceleration Structure (TLAS) from a list of BVH instances.

                Args:
                    instances (numpy.ndarray): A structured array with `instance_dtype` describing
                                               each instance's transform, blas_id, and mask.
                    BLASes (List[BVH]): A list of the BVH objects to be instanced. The `blas_id`
                                        in the instances array corresponds to the index in this list.

                Returns:
                    BVH: A new BVH instance representing the TLAS.
            ))")

        .def_static("load", [](
            const py::object& filepath_obj,
            py::array_t<float, py::array::c_style | py::array::forcecast> vertices_np,
            py::object indices_obj // py::object to accept an array or None
        ) {
            if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
                throw std::runtime_error("Vertex data for loading must be a 2D numpy array with shape (V, 4).");
            }

            std::string filepath = py::str(filepath_obj).cast<std::string>();
            auto wrapper = std::make_unique<PyBVH>();
            bool success = false;

            if (indices_obj.is_none()) {

                // Case: non-indexed geometry
                wrapper->source_geometry_refs.append(vertices_np);

                py::buffer_info vertices_buf = vertices_np.request();
                auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(vertices_buf.ptr);
                const uint32_t prim_count = static_cast<uint32_t>(vertices_buf.shape[0] / 3);

                success = wrapper->bvh->Load(filepath.c_str(), vertices_ptr, prim_count);
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
                const uint32_t prim_count = static_cast<uint32_t>(indices_buf.shape[0]);
                success = wrapper->bvh->Load(filepath.c_str(), vertices_ptr, indices_ptr, prim_count);
            }

            if (!success) {
                throw std::runtime_error(
                    "Failed to load BVH from file. Check file integrity, version, and that "
                    "you provided the correct geometry layout (indexed vs. non-indexed)."
                );
            }

            // Infer quality from refittable flag since it is not stored in the file
            wrapper->quality = wrapper->bvh->refittable ? BuildQuality::Balanced : BuildQuality::High;

            return wrapper;

        }, py::arg("filepath"), py::arg("vertices").noconvert(), py::arg("indices").noconvert() = py::none(),
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
            ))")

        .def("save", [](const PyBVH& self, const py::object& filepath_obj) {
            std::string filepath = py::str(filepath_obj).cast<std::string>();
            self.bvh->Save(filepath.c_str());
        }, py::arg("filepath"),
            R"((
                Saves the BVH to a file.

                Args:
                    filepath (str or pathlib.Path): The path where the BVH file will be saved.
            ))"
        )

        .def_property_readonly("nodes", [](PyBVH &self) -> py::array {
            if (!self.bvh || self.bvh->usedNodes == 0) {
                py::dtype dt = py::dtype::of<tinybvh::BVH::BVHNode>();
                return py::array(dt, {0}, {});
            }
            return py::array_t<tinybvh::BVH::BVHNode>(
                { (py::ssize_t)self.bvh->usedNodes }, self.bvh->bvhNode, py::cast(self)
            );
        }, "The structured numpy array of BVH nodes.")

        .def_property_readonly("prim_indices", [](PyBVH &self) -> py::array {
            if (!self.bvh || self.bvh->idxCount == 0) {
                return py::array_t<uint32_t>(0);
            }
            return py::array_t<uint32_t>(
                { (py::ssize_t)self.bvh->idxCount }, // shape
                self.bvh->primIdx,    // data pointer
                py::cast(self)        // owner
            );
        }, "The array of primitive indices, ordered for locality.")

        .def_property_readonly("quality", [](const PyBVH &self) { return self.quality; },
            "The build quality level used to construct the BVH.")

         .def("intersect", [](PyBVH &self, PyRay &py_ray) -> float {

            if (self.bvh->triCount == 0) {
                return INFINITY; // empty BVH, nothing to intersect
            }

            if (self.custom_type == PyBVH::CustomType::AABB) {

                auto aabbs_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                auto inv_extents_np = py::cast<py::array_t<float>>(self.source_geometry_refs[1]);
                g_aabbs_ptr = aabbs_np.data();
                g_inv_extents_ptr = inv_extents_np.data();

            } else if (self.custom_type == PyBVH::CustomType::Sphere) {

                auto points_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_points_ptr = points_np.data();
                g_sphere_radius = self.sphere_radius;
            }

            tinybvh::Ray ray(py_ray.origin, py_ray.direction, py_ray.t);
            ray.mask = py_ray.mask;

            self.bvh->Intersect(ray);

            // unset pointers
            g_aabbs_ptr = nullptr;
            g_points_ptr = nullptr;
            g_inv_extents_ptr = nullptr;

            if (ray.hit.t < py_ray.t) {
                // update python Ray with hit information
                py_ray.t = ray.hit.t;
                py_ray.u = ray.hit.u;
                py_ray.v = ray.hit.v;
                py_ray.prim_id = ray.hit.prim;

                #if INST_IDX_BITS == 32
                    py_ray.prim_id = ray.hit.prim;
                    py_ray.inst_id = ray.hit.inst;
                #else
                    // unpack combined ID
                    py_ray.inst_id = ray.hit.prim >> INST_IDX_SHFT;
                    py_ray.prim_id = ray.hit.prim & PRIM_IDX_MASK;
                #endif

                return ray.hit.t;
            }
            // no hit found (or no closer hit). Return infinity.
            return INFINITY;
        }, py::arg("ray"),
           R"((
                Performs an intersection query with a single ray.

                This method modifies the passed Ray object in-place if a closer hit is found.

                Args:
                    ray (Ray): The ray to test. Its `t`, `u`, `v`, and `prim_id` attributes
                               will be updated upon a successful hit.

                Returns:
                    float: The hit distance `t` if a hit was found, otherwise `infinity`.
           ))")

        .def("intersect_batch", [](PyBVH &self,
                           py::array_t<float, py::array::c_style | py::array::forcecast> origins_np,
                           py::array_t<float, py::array::c_style | py::array::forcecast> directions_np,
                           py::object t_max_obj,
                           py::object masks_obj) -> py::array
            {
                // Validation
                if (origins_np.ndim() != 2 || origins_np.shape(1) != 3) {
                    throw std::runtime_error("Origins must be a 2D numpy array with shape (N, 3).");
                }
                if (directions_np.ndim() != 2 || directions_np.shape(1) != 3) {
                    throw std::runtime_error("Directions must be a 2D numpy array with shape (N, 3).");
                }
                if (origins_np.shape(0) != directions_np.shape(0)) {
                    throw std::runtime_error("Origins and directions arrays must have the same number of rows.");
                }
            
                const py::ssize_t n_rays = origins_np.shape(0);
                if (n_rays == 0) return py::array_t<HitRecord>(0);
            
                // handle empty BVH
                if (self.bvh->triCount == 0) {
                    auto result_np = py::array_t<HitRecord>(n_rays);
                    auto* result_ptr = result_np.mutable_data();
                    for (py::ssize_t i = 0; i < n_rays; ++i) {
                        result_ptr[i] = {static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), INFINITY, 0.0f, 0.0f};
                    }
                    return result_np;
                }
            
                // Prepare input data pointers
                auto origins_ptr = origins_np.data();
                auto directions_ptr = directions_np.data();
            
                // handle optional t_max
                const float* t_max_ptr = nullptr;
                py::array_t<float, py::array::c_style | py::array::forcecast> t_max_np;
                if (!t_max_obj.is_none()) {
                    t_max_np = py::cast<py::array_t<float, py::array::c_style>>(t_max_obj);
                    if (t_max_np.ndim() != 1 || t_max_np.shape(0) != n_rays) throw std::runtime_error("t_max must be a 1D array with length N.");
                    t_max_ptr = t_max_np.data();
                }
            
                // handle optional masks
                const uint32_t* masks_ptr = nullptr;
                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> masks_np;
                if (!masks_obj.is_none()) {
                    masks_np = py::cast<py::array_t<uint32_t, py::array::c_style>>(masks_obj);
                    if (masks_np.ndim() != 1 || masks_np.shape(0) != n_rays) {
                        throw std::runtime_error("masks must be a 1D uint32 array with length N.");
                    }
                    masks_ptr = masks_np.data();
                }
            
                // Prepare output and Ray vector ---
                auto result_np = py::array_t<HitRecord>(n_rays);
                auto* result_ptr = result_np.mutable_data();
                std::vector<tinybvh::Ray> rays(n_rays);
            
                // Decide which path to take before releasing the GIL.
                bool use_parallel = (self.custom_type == PyBVH::CustomType::None);
            
                const float* aabbs_data_ptr = nullptr;
                const float* inv_extents_data_ptr = nullptr;
                const float* points_data_ptr = nullptr;
                float sphere_rad = 0.0f;
                
                // If using serial path, get raw C++ pointers while we still have the GIL.
                if (!use_parallel) {
                    if (self.custom_type == PyBVH::CustomType::AABB) {
                        aabbs_data_ptr = py::cast<py::array_t<float>>(self.source_geometry_refs[0]).data();
                        inv_extents_data_ptr = py::cast<py::array_t<float>>(self.source_geometry_refs[1]).data();
                    } else if (self.custom_type == PyBVH::CustomType::Sphere) {
                        points_data_ptr = py::cast<py::array_t<float>>(self.source_geometry_refs[0]).data();
                        sphere_rad = self.sphere_radius;
                    }
                }
            
                {
                    py::gil_scoped_release release;  // Release GIL for C++ threading
            
                    // initialize rays in parallel (this is GIL-safe)
                    #ifdef _OPENMP
                        #pragma omp parallel for schedule(static)
                    #endif
                    for (py::ssize_t i = 0; i < n_rays; ++i) {
                        const size_t i3 = i * 3;
                        const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                        const uint32_t mask_init = masks_ptr ? masks_ptr[i] : 0xFFFF; // default to all
                        rays[i] = tinybvh::Ray(
                            {origins_ptr[i3], origins_ptr[i3+1], origins_ptr[i3+2]},
                            {directions_ptr[i3], directions_ptr[i3+1], directions_ptr[i3+2]},
                            t_init,
                            mask_init
                        );
                    }
                    
                    if (use_parallel) {
                        // Fast path: Standard triangles, using Intersect256Rays
                        #ifdef _OPENMP
                            #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; i += 256) {
                            const py::ssize_t end = std::min(i + 256, n_rays);
                            const py::ssize_t count = end - i;
                            if (count == 256) {
                                #ifdef BVH_USEAVX
                                    self.bvh->Intersect256RaysSSE(&rays[i]);
                                #else
                                    self.bvh->Intersect256Rays(&rays[i]);
                                #endif
                            } else {
                                for (py::ssize_t j = i; j < end; ++j) self.bvh->Intersect(rays[j]);
                            }
                        }
                    } else {
                        // Slow path for custom geometry (process serially to handle thread_local data)
                        g_aabbs_ptr = aabbs_data_ptr;
                        g_inv_extents_ptr = inv_extents_data_ptr;
                        g_points_ptr = points_data_ptr;
                        g_sphere_radius = sphere_rad;
            
                        for (py::ssize_t i = 0; i < n_rays; ++i) {
                            self.bvh->Intersect(rays[i]);
                        }
            
                        // unset global pointers
                        g_aabbs_ptr = nullptr;
                        g_inv_extents_ptr = nullptr;
                        g_points_ptr = nullptr;
                    }
                } // GIL re-acquired here
            
                // Copy results back to numpy
                for (py::ssize_t i = 0; i < n_rays; ++i) {
                    const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                    if (rays[i].hit.t < t_init) {
                        #if INST_IDX_BITS == 32
                            result_ptr[i] = {rays[i].hit.prim, rays[i].hit.inst, rays[i].hit.t, rays[i].hit.u, rays[i].hit.v};
                        #else
                            // if instance ID is packed in prim_id, unpack it
                            uint32_t inst_id = rays[i].hit.prim >> INST_IDX_SHFT;
                            uint32_t prim_id = rays[i].hit.prim & PRIM_IDX_MASK;
                            result_ptr[i] = {prim_id, inst_id, rays[i].hit.t, rays[i].hit.u, rays[i].hit.v};
                        #endif
                    } else {
                        result_ptr[i] = {static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), INFINITY, 0.0f, 0.0f};
                    }
                }
                return result_np;
            
            }, py::arg("origins"), py::arg("directions"), py::arg("t_max") = py::none(), py::arg("masks") = py::none(),
               R"((
                    Performs intersection queries for a batch of rays.
            
                    This method leverages both multi-core processing (via OpenMP) and SIMD instructions
                    (via tinybvh's Intersect256Rays functions) for maximum throughput on standard
                    triangle meshes. For custom geometry like AABBs or spheres, it falls back to a
                    serial implementation.
            
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
               ))")

        .def("is_occluded", [](PyBVH &self, const PyRay &py_ray) -> bool {

            if (self.bvh->triCount == 0) {
                return false; // Nothing to occlude
            }

            if (self.custom_type == PyBVH::CustomType::AABB) {
                auto aabbs_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_aabbs_ptr = aabbs_np.data();
            } else if (self.custom_type == PyBVH::CustomType::Sphere) {
                auto points_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_points_ptr = points_np.data();
                g_sphere_radius = self.sphere_radius;
            }

            tinybvh::Ray ray(
                py_ray.origin,
                py_ray.direction,
                py_ray.t);  // The ray's t is used as the maximum distance for the occlusion check

            ray.mask = py_ray.mask;

            bool result = self.bvh->IsOccluded(ray);

            // unset pointers
            g_aabbs_ptr = nullptr;
            g_points_ptr = nullptr;

             return result;

        }, py::arg("ray"),
           R"((
                Performs an occlusion query with a single ray.

                Checks if any geometry is hit by the ray within the distance specified by `ray.t`.
                This is typically faster than `intersect` as it can stop at the first hit.

                Args:
                    ray (Ray): The ray to test.

                Returns:
                    bool: True if the ray is occluded, False otherwise.
           ))")

        .def("is_occluded_batch", [](PyBVH &self, py::array_t<float, py::array::c_style> origins_np,
                                       py::array_t<float, py::array::c_style> directions_np,
                                       py::object t_max_obj,
                                       py::object masks_obj) -> py::array_t<bool>
            {
                // input validation
                if (origins_np.ndim() != 2 || origins_np.shape(1) != 3) {
                    throw std::runtime_error("Origins must be a 2D numpy array with shape (N, 3).");
                }
                if (directions_np.ndim() != 2 || directions_np.shape(1) != 3) {
                    throw std::runtime_error("Directions must be a 2D numpy array with shape (N, 3).");
                }
                if (origins_np.shape(0) != directions_np.shape(0)) {
                    throw std::runtime_error("Origins and directions arrays must have the same number of rows.");
                }

                const py::ssize_t n_rays = origins_np.shape(0);
                if (n_rays == 0) return py::array_t<bool>(0);

                // check for empty BVH
                if (self.bvh->triCount == 0) {
                    auto result_np = py::array_t<bool>(n_rays);
                    std::fill(result_np.mutable_data(), result_np.mutable_data() + n_rays, false);
                    return result_np;
                }

                auto origins_ptr = origins_np.data();
                auto directions_ptr = directions_np.data();

                const float* t_max_ptr = nullptr;
                py::array_t<float, py::array::c_style | py::array::forcecast> t_max_np;
                if (!t_max_obj.is_none()) {
                    t_max_np = py::cast<py::array_t<float, py::array::c_style>>(t_max_obj);
                    if (t_max_np.ndim() != 1 || t_max_np.shape(0) != n_rays) throw std::runtime_error("t_max must be a 1D array with length N.");
                    t_max_ptr = t_max_np.data();
                }

                const uint32_t* masks_ptr = nullptr;
                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> masks_np;
                if (!masks_obj.is_none()) {
                    masks_np = py::cast<py::array_t<uint32_t, py::array::c_style>>(masks_obj);
                    if (masks_np.ndim() != 1 || masks_np.shape(0) != n_rays) {
                        throw std::runtime_error("masks must be a 1D uint32 array with length N.");
                    }
                    masks_ptr = masks_np.data();
                }

                auto result_np = py::array_t<bool>(n_rays);
                auto* result_ptr = result_np.mutable_data();

                // Decide which path to take before releasing the GIL.
                bool use_parallel = (self.custom_type == PyBVH::CustomType::None);

                const float* aabbs_data_ptr = nullptr;
                const float* points_data_ptr = nullptr;
                float sphere_rad = 0.0f;

                // If using serial path, get raw C++ pointers while we still have the GIL.
                if (!use_parallel) {
                    if (self.custom_type == PyBVH::CustomType::AABB) {
                        aabbs_data_ptr = py::cast<py::array_t<float>>(self.source_geometry_refs[0]).data();
                    } else if (self.custom_type == PyBVH::CustomType::Sphere) {
                        points_data_ptr = py::cast<py::array_t<float>>(self.source_geometry_refs[0]).data();
                        sphere_rad = self.sphere_radius;
                    }
                }

                {
                    py::gil_scoped_release release; // Release GIL for C++ threading

                    if (use_parallel) {
                        // Fast path: Standard triangles, parallelized with OpenMP.
                        #ifdef _OPENMP
                            #pragma omp parallel for schedule(dynamic)
                        #endif
                        for (py::ssize_t i = 0; i < n_rays; ++i) {
                            const size_t i3 = i * 3;
                            const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                            const uint32_t mask_init = masks_ptr ? masks_ptr[i] : 0xFFFF;

                            tinybvh::Ray ray(
                                {origins_ptr[i3], origins_ptr[i3+1], origins_ptr[i3+2]},
                                {directions_ptr[i3], directions_ptr[i3+1], directions_ptr[i3+2]},
                                t_init,
                                mask_init
                            );
                            result_ptr[i] = self.bvh->IsOccluded(ray);
                        }
                    } else {
                        // Slow path: Custom geometry, processed serially to safely use thread_local pointers.
                        g_aabbs_ptr = aabbs_data_ptr;
                        g_points_ptr = points_data_ptr;
                        g_sphere_radius = sphere_rad;

                        for (py::ssize_t i = 0; i < n_rays; ++i) {
                            const size_t i3 = i * 3;
                            const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;
                            const uint32_t mask_init = masks_ptr ? masks_ptr[i] : 0xFFFF;

                            tinybvh::Ray ray(
                                {origins_ptr[i3], origins_ptr[i3+1], origins_ptr[i3+2]},
                                {directions_ptr[i3], directions_ptr[i3+1], directions_ptr[i3+2]},
                                t_init,
                                mask_init
                            );
                            result_ptr[i] = self.bvh->IsOccluded(ray);
                        }

                        // unset pointers now that the loop is done.
                        g_aabbs_ptr = nullptr;
                        g_points_ptr = nullptr;
                    }
                } // GIL re-acquired

                return result_np;

            }, py::arg("origins"), py::arg("directions"), py::arg("t_max") = py::none(), py::arg("masks") = py::none(),
               R"((
                    Performs occlusion queries for a batch of rays, parallelized for performance.

                    This method leverages multi-core processing (via OpenMP) for maximum throughput on
                    standard triangle meshes. For custom geometry, it falls back to a serial implementation
                    to ensure thread-safety.

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
               ))")

        .def("refit", &PyBVH::refit,
             R"((
                Refits the BVH to the current state of the source geometry, which is much faster than a full rebuild.

                Should be called after the underlying vertex data (numpy array used for construction)
                has been modified.

                Note: This will fail if the BVH was built with spatial splits (with BuildQuality.High).
             ))")

        .def_property_readonly("node_count", [](const PyBVH &self){ return self.bvh->usedNodes; }, "Total number of nodes in the BVH.")
        .def_property_readonly("prim_count", [](const PyBVH &self){ return self.bvh->triCount; }, "Total number of primitives in the BVH.")
        .def_property_readonly("aabb_min", [](PyBVH &self) -> py::array {
            if (!self.bvh) return py::array_t<float>(0);
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMin data
            return py::array_t<float>(
                {3},                                // shape
                {&self.bvh->aabbMin.x},             // data pointer
                py::cast(self)                      // owner
            );
        }, "The minimum corner of the root axis-aligned bounding box.")
        .def_property_readonly("aabb_max", [](PyBVH &self) -> py::array {
            if (!self.bvh) return py::array_t<float>(0);
            // Create a 1D array of shape (3,) that is a view into the bvh.aabbMax data
            return py::array_t<float>(
                {3},                                // shape
                {&self.bvh->aabbMax.x},             // data pointer
                py::cast(self)                      // owner
            );
        }, "The maximum corner of the root axis-aligned bounding box.");
}