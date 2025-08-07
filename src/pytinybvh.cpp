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

    float t = tinybvh_intersect_aabb(ray, bmin, bmax);

    if (t < ray.hit.t) {
        ray.hit.t = t;
        ray.hit.prim = prim_id;
        ray.hit.u = 0.0f; // No barycentric coords for AABB
        ray.hit.v = 0.0f;
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

    // Inlined slab test
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
        float t = (-b - sqrt(discriminant)) / (2.0f * a);
        if (t > 1e-6f && t < ray.hit.t) { // Check for valid t range
            ray.hit.t = t;
            ray.hit.prim = prim_id;
            ray.hit.u = 0.0f; // No barycentric coords for sphere
            ray.hit.v = 0.0f;
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

    const float t = (-b - sqrt(discriminant)) / (2.0f * a);
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

    BuildQuality quality = BuildQuality::Balanced;

    enum class CustomType { None, AABB, Sphere };
    CustomType custom_type = CustomType::None;
    float sphere_radius = 0.0f;

    PyBVH() : bvh(std::make_unique<tinybvh::BVH>()) {}

    // Core builders (zero-copy)

    static std::unique_ptr<PyBVH> from_vertices(py::array_t<float, py::array::c_style | py::array::forcecast> vertices_np, BuildQuality quality) {
        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (M, 4).");
        }
        if (vertices_np.shape(0) % 3 != 0) {
            throw std::runtime_error("Input vertex count must be a multiple of 3.");
        }

        auto wrapper = std::make_unique<PyBVH>();

        wrapper->source_geometry_refs.append(vertices_np);   // reference to the numpy array
        wrapper->quality = quality;

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
        py::array_t<float, py::array::c_style | py::array::forcecast> vertices_np,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> indices_np,
        BuildQuality quality)
    {
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

        py::buffer_info aabbs_buf = aabbs_np.request();
        const auto* aabbs_ptr = static_cast<const float*>(aabbs_buf.ptr);
        const uint32_t prim_count = static_cast<uint32_t>(aabbs_buf.shape[0]);

        // thread-local pointer before calling Build
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

    // Convenience builders (with copying)

    static std::unique_ptr<PyBVH> from_triangles(py::array_t<float, py::array::c_style | py::array::forcecast> tris_np, BuildQuality quality) {
        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }

        // this must be done inside this function so vertices_np stays alive
        auto wrapper = std::make_unique<PyBVH>();
        wrapper->quality = quality;

        const size_t num_tris = tris_np.shape(0);

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
        wrapper->source_geometry_refs.append(points_np); // Store original points
        wrapper->quality = quality;
        wrapper->custom_type = CustomType::Sphere;
        wrapper->sphere_radius = radius;

        wrapper->bvh->customIntersect = sphere_intersect_callback;
        wrapper->bvh->customIsOccluded = sphere_isoccluded_callback;

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

    void refit() {
        if (!bvh || !bvh->refittable) {
            throw std::runtime_error("BVH is not refittable (it might have spatial splits, i.e. built with BuildQuality.High).");
        }
        // Refit needs original vertex data
        // `bvh->verts` slice already points to it
        bvh->Refit();
    }
};

// pybind11 wrapper for tinybvh::Ray to be used in intersection queries
struct PyRay {
    tinybvh::bvhvec3 origin;
    tinybvh::bvhvec3 direction;
    float t = 1e30f;
    float u = 0.0f, v = 0.0f;
    uint32_t prim_id = -1;
};


// =====================================================================================================================


PYBIND11_MODULE(pytinybvh, m) {
    m.doc() = "Python bindings for the tinybvh library";

    // HitRecord struct and its numpy dtype
    struct HitRecord {
        uint32_t prim_id;
        float t, u, v;
    };
    PYBIND11_NUMPY_DTYPE(HitRecord, prim_id, t, u, v);


    PYBIND11_NUMPY_DTYPE(tinybvh::bvhvec3, x, y, z);
    PYBIND11_NUMPY_DTYPE_EX(tinybvh::BVH::BVHNode, aabbMin, "aabb_min", leftFirst, "left_first", aabbMax, "aabb_max", triCount, "prim_count");

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
        .def(py::init([](const py::object& origin_obj, const py::object& direction_obj, float t) {
                 return PyRay{py_obj_to_vec3(origin_obj), py_obj_to_vec3(direction_obj), t, 0.f, 0.f, (uint32_t)-1};
             }), "origin"_a, "direction"_a, "t"_a = 1e30f)
        .def_property("origin",
            [](const PyRay &r) { return py::make_tuple(r.origin.x, r.origin.y, r.origin.z); },
            [](PyRay &r, const py::object& obj) { r.origin = py_obj_to_vec3(obj); },
            "The origin point of the ray (list, tuple, or numpy array).")
        .def_property("direction",
            [](const PyRay &r) { return py::make_tuple(r.direction.x, r.direction.y, r.direction.z); },
            [](PyRay &r, const py::object& obj) { r.direction = py_obj_to_vec3(obj); },
            "The direction vector of the ray (list, tuple, or numpy array).")
        .def_readwrite("t", &PyRay::t, "The maximum distance for intersection. Updated with hit distance.")
        .def_readonly("u", &PyRay::u, "Barycentric u-coordinate of the hit.")
        .def_readonly("v", &PyRay::v, "Barycentric v-coordinate of the hit.")
        .def_readonly("prim_id", &PyRay::prim_id, "The ID of the primitive that was hit.")
        .def("__repr__", [](const PyRay &r) {
            if (r.prim_id != (uint32_t)-1) {
                return "<pytinybvh.Ray hit primitive " + std::to_string(r.prim_id) + " at t=" + std::to_string(r.t) + ">";
            }
            return std::string("<pytinybvh.Ray miss>");
        });


    py::class_<PyBVH>(m, "BVH", "A Bounding Volume Hierarchy for fast ray intersections.")

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
            py::arg("triangles").noconvert(), py::arg("quality") = BuildQuality::Balanced)

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
            py::arg("points").noconvert(), py::arg("radius") = 1e-5f, py::arg("quality") = BuildQuality::Balanced)

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
            py::arg("vertices").noconvert(), py::arg("indices").noconvert(), py::arg("quality") = BuildQuality::Balanced)

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
            py::arg("aabbs").noconvert(), py::arg("quality") = BuildQuality::Balanced)

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

            if (self.custom_type == PyBVH::CustomType::AABB) {
                auto aabbs_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_aabbs_ptr = aabbs_np.data();
            } else if (self.custom_type == PyBVH::CustomType::Sphere) {
                auto points_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_points_ptr = points_np.data();
                g_sphere_radius = self.sphere_radius;
            }

            tinybvh::Ray ray(py_ray.origin, py_ray.direction, py_ray.t);
            self.bvh->Intersect(ray);

            // unset pointers
            g_aabbs_ptr = nullptr;
            g_points_ptr = nullptr;

            if (ray.hit.t < py_ray.t) {
                // update python Ray with hit information
                py_ray.t = ray.hit.t;
                py_ray.u = ray.hit.u;
                py_ray.v = ray.hit.v;
                py_ray.prim_id = ray.hit.prim;
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

        .def("intersect_batch", [](PyBVH &self, py::array_t<float, py::array::c_style> origins_np,
                                   py::array_t<float, py::array::c_style> directions_np,
                                   py::object t_max_obj) -> py::array
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

            // access the numpy array buffers
            auto origins_buf = origins_np.request();
            auto directions_buf = directions_np.request();
            const size_t n_rays = origins_buf.shape[0];
            const auto* origins_ptr = static_cast<const float*>(origins_buf.ptr);
            const auto* directions_ptr = static_cast<const float*>(directions_buf.ptr);

            // handle optional t_max array
            const float* t_max_ptr = nullptr;
            py::array_t<float, py::array::c_style> t_max_np;
            if (!t_max_obj.is_none()) {
                t_max_np = py::cast<py::array_t<float, py::array::c_style>>(t_max_obj);
                if (t_max_np.ndim() != 1 || t_max_np.shape(0) != n_rays) {
                    throw std::runtime_error("t_max must be a 1D array with length N.");
                }
                t_max_ptr = t_max_np.data();
            }

            // Create output structured numpy array
            auto result_np = py::array_t<HitRecord>(n_rays);
            auto result_buf = result_np.request();
            auto* result_ptr = static_cast<HitRecord*>(result_buf.ptr);

            if (self.custom_type == PyBVH::CustomType::AABB) {
                auto aabbs_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_aabbs_ptr = aabbs_np.data();
            } else if (self.custom_type == PyBVH::CustomType::Sphere) {
                auto points_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_points_ptr = points_np.data();
                g_sphere_radius = self.sphere_radius;
            }

            for (size_t i = 0; i < n_rays; ++i) {
                const size_t i3 = i * 3;
                const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;

                tinybvh::Ray ray(
                    tinybvh::bvhvec3(origins_ptr[i3], origins_ptr[i3+1], origins_ptr[i3+2]),
                    tinybvh::bvhvec3(directions_ptr[i3], directions_ptr[i3+1], directions_ptr[i3+2]),
                    t_init
                );

                self.bvh->Intersect(ray);

                if (ray.hit.t < t_init) {
                    result_ptr[i] = {ray.hit.prim, ray.hit.t, ray.hit.u, ray.hit.v};
                } else {
                    // sentinel value for a miss is all the bits to 1
                    result_ptr[i] = {static_cast<uint32_t>(-1), INFINITY, 0.0f, 0.0f};
                }
            }

            // unset pointers
            g_aabbs_ptr = nullptr;
            g_points_ptr = nullptr;

            return result_np;

        }, py::arg("origins").noconvert(), py::arg("directions").noconvert(), py::arg("t_max") = py::none(),
           R"((
                Performs intersection queries for a batch of rays.

                Args:
                    origins (numpy.ndarray): A (N, 3) float array of ray origins.
                    directions (numpy.ndarray): A (N, 3) float array of ray directions.
                    t_max (numpy.ndarray, optional): A (N,) float array of maximum intersection distances.

                Returns:
                    numpy.ndarray: A structured array of shape (N,) with dtype
                                   [('prim_id', '<u4'), ('t', '<f4'), ('u', '<f4'), ('v', '<f4')].
                                   For misses, prim_id is -1 and t is infinity.
           ))")

        .def("is_occluded", [](PyBVH &self, const PyRay &py_ray) -> bool {

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
                                       py::object t_max_obj) -> py::array_t<bool>
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

            auto origins_buf = origins_np.request();
            auto directions_buf = directions_np.request();
            const size_t n_rays = origins_buf.shape[0];
            const auto* origins_ptr = static_cast<const float*>(origins_buf.ptr);
            const auto* directions_ptr = static_cast<const float*>(directions_buf.ptr);

            const float* t_max_ptr = nullptr;
            py::array_t<float, py::array::c_style> t_max_np;
            if (!t_max_obj.is_none()) {
                t_max_np = py::cast<py::array_t<float, py::array::c_style>>(t_max_obj);
                if (t_max_np.ndim() != 1 || t_max_np.shape(0) != n_rays) {
                    throw std::runtime_error("t_max must be a 1D array with length N.");
                }
                t_max_ptr = t_max_np.data();
            }

            // Create boolean output array
            auto result_np = py::array_t<bool>(n_rays);
            auto result_buf = result_np.request();
            auto* result_ptr = static_cast<bool*>(result_buf.ptr);

// TODO: This might work fine too, but maybe less explicit, idk
//            const size_t n_rays = origins_np.shape(0);
//            const auto* origins_ptr = origins_np.data();
//            const auto* directions_ptr = directions_np.data();


            if (self.custom_type == PyBVH::CustomType::AABB) {
                auto aabbs_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_aabbs_ptr = aabbs_np.data();
            } else if (self.custom_type == PyBVH::CustomType::Sphere) {
                auto points_np = py::cast<py::array_t<float>>(self.source_geometry_refs[0]);
                g_points_ptr = points_np.data();
                g_sphere_radius = self.sphere_radius;
            }

            for (size_t i = 0; i < n_rays; ++i) {
                const size_t i3 = i * 3;
                const float t_init = t_max_ptr ? t_max_ptr[i] : 1e30f;

                tinybvh::Ray ray(
                    tinybvh::bvhvec3(origins_ptr[i3], origins_ptr[i3+1], origins_ptr[i3+2]),
                    tinybvh::bvhvec3(directions_ptr[i3], directions_ptr[i3+1], directions_ptr[i3+2]),
                    t_init
                );

                result_ptr[i] = self.bvh->IsOccluded(ray);
            }

            // unset pointers
            g_aabbs_ptr = nullptr;
            g_points_ptr = nullptr;

            return result_np;

        }, py::arg("origins").noconvert(), py::arg("directions").noconvert(), py::arg("t_max") = py::none(),
           R"((
                Performs occlusion queries for a batch of rays.

                Args:
                    origins (numpy.ndarray): A (N, 3) float array of ray origins.
                    directions (numpy.ndarray): A (N, 3) float array of ray directions.
                    t_max (numpy.ndarray, optional): A (N,) float array of maximum occlusion distances.
                                                     If a hit is found beyond this distance, it is ignored.

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