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

// mini 3D vector struct to avoid having a dependency on glm just for 3 floats (might change in the future tho)
struct Vector3f {
    float x = 0.f, y = 0.f, z = 0.f;

    Vector3f() = default;
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Enum for build quality selection
enum class BuildQuality {
    Quick,
    Balanced,
    High
};

// C++ class to wrap the tinybvh::BVH object and its associated data: this is the object held by the Python BVH class
struct BVHWrapper {
    std::unique_ptr<tinybvh::BVH> bvh;

    py::array source_geometry; // ref to the Python numpy array that contains the source vertices
    // This prevents Python's garbage collector from freeing the memory while the C++ BVH object might still be using it
    // (that's important especially for refitting)

    BuildQuality quality = BuildQuality::Balanced;

    BVHWrapper() : bvh(std::make_unique<tinybvh::BVH>()) {}

    static std::unique_ptr<BVHWrapper> from_vertices_4f(py::array_t<float, py::array::c_style | py::array::forcecast> vertices_np, BuildQuality quality) {
        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (M, 4).");
        }
        if (vertices_np.shape(0) % 3 != 0) {
            throw std::runtime_error("Input vertex count must be a multiple of 3.");
        }

        auto wrapper = std::make_unique<BVHWrapper>();
        // no copy: ref to the numpy array
        wrapper->source_geometry = vertices_np;

        wrapper->quality = quality;

        py::buffer_info buf = vertices_np.request();
        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(buf.ptr);
        size_t prim_count = buf.shape[0] / 3;

        switch (quality) {
            case BuildQuality::Quick:
                wrapper->bvh->BuildQuick(vertices_ptr, static_cast<uint32_t>(prim_count));
                break;
            case BuildQuality::High:
                wrapper->bvh->BuildHQ(vertices_ptr, static_cast<uint32_t>(prim_count));
                break;
            case BuildQuality::Balanced:
            default:
                wrapper->bvh->Build(vertices_ptr, static_cast<uint32_t>(prim_count));
                break;
        }

        return wrapper;
    }

    static std::unique_ptr<BVHWrapper> from_triangles(py::array_t<float, py::array::c_style | py::array::forcecast> tris_np, BuildQuality quality) {
        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }

        auto wrapper = std::make_unique<BVHWrapper>();
        wrapper->quality = quality;

        const size_t num_tris = tris_np.shape(0);
        const float* tris_ptr = tris_np.data();

        // This copy is necessary to bridge the Python friendly (N, 9) layout to tinybvh's layout
        auto vertices_np = py::array_t<float>(py::array::ShapeContainer({(py::ssize_t)num_tris * 3, 4}));
        auto v_buf = vertices_np.request();
        float* v_ptr = static_cast<float*>(v_buf.ptr);

        for (size_t i = 0; i < num_tris; ++i) {
            // Triangle i, vertex 0
            v_ptr[(i * 3 + 0) * 4 + 0] = tris_ptr[i * 9 + 0];
            v_ptr[(i * 3 + 0) * 4 + 1] = tris_ptr[i * 9 + 1];
            v_ptr[(i * 3 + 0) * 4 + 2] = tris_ptr[i * 9 + 2];
            v_ptr[(i * 3 + 0) * 4 + 3] = 0.0f; // Dummy w
            // Triangle i, vertex 1
            v_ptr[(i * 3 + 1) * 4 + 0] = tris_ptr[i * 9 + 3];
            v_ptr[(i * 3 + 1) * 4 + 1] = tris_ptr[i * 9 + 4];
            v_ptr[(i * 3 + 1) * 4 + 2] = tris_ptr[i * 9 + 5];
            v_ptr[(i * 3 + 1) * 4 + 3] = 0.0f; // Dummy w
            // Triangle i, vertex 2
            v_ptr[(i * 3 + 2) * 4 + 0] = tris_ptr[i * 9 + 6];
            v_ptr[(i * 3 + 2) * 4 + 1] = tris_ptr[i * 9 + 7];
            v_ptr[(i * 3 + 2) * 4 + 2] = tris_ptr[i * 9 + 8];
            v_ptr[(i * 3 + 2) * 4 + 3] = 0.0f; // Dummy w
        }

        wrapper->source_geometry = vertices_np; // keep formatted geometry alive

        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(v_buf.ptr);

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

    static std::unique_ptr<BVHWrapper> from_points(py::array_t<float, py::array::c_style | py::array::forcecast> points_np, float radius, BuildQuality quality) {
        if (points_np.ndim() != 2 || points_np.shape(1) != 3) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 3).");
        }
        if (radius <= 0.0f) {
            throw std::runtime_error("Point radius must be positive.");
        }

        auto wrapper = std::make_unique<BVHWrapper>();
        wrapper->quality = quality;

        size_t num_points = points_np.shape(0);
        const float* ptr = points_np.data();

        auto vertices_np = py::array_t<float>(py::array::ShapeContainer({(py::ssize_t)num_points * 3, 4}));
        float* v_ptr = vertices_np.mutable_data();

        for (size_t i = 0; i < num_points; ++i) {
            float x = ptr[i * 3 + 0];
            float y = ptr[i * 3 + 1];
            float z = ptr[i * 3 + 2];
            // v0: min corner
            v_ptr[(i * 3 + 0) * 4 + 0] = x - radius;
            v_ptr[(i * 3 + 0) * 4 + 1] = y - radius;
            v_ptr[(i * 3 + 0) * 4 + 2] = z - radius;
            v_ptr[(i * 3 + 0) * 4 + 3] = 0.0f;
            // v1: max corner
            v_ptr[(i * 3 + 1) * 4 + 0] = x + radius;
            v_ptr[(i * 3 + 1) * 4 + 1] = y + radius;
            v_ptr[(i * 3 + 1) * 4 + 2] = z + radius;
            v_ptr[(i * 3 + 1) * 4 + 3] = 0.0f;
            // v2: repeat min corner
            v_ptr[(i * 3 + 2) * 4 + 0] = x - radius;
            v_ptr[(i * 3 + 2) * 4 + 1] = y - radius; v_ptr[(i * 3 + 2) * 4 + 2] = z - radius;
            v_ptr[(i * 3 + 2) * 4 + 3] = 0.0f;
        }

        wrapper->source_geometry = vertices_np; // keep formatted geometry alive

        py::buffer_info buf = vertices_np.request();
        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(buf.ptr);

        switch (quality) {
            case BuildQuality::Quick:
                wrapper->bvh->BuildQuick(vertices_ptr, static_cast<uint32_t>(num_points));
                break;
            case BuildQuality::High:
                wrapper->bvh->BuildHQ(vertices_ptr, static_cast<uint32_t>(num_points));
                break;
            case BuildQuality::Balanced:
            default:
                wrapper->bvh->Build(vertices_ptr, static_cast<uint32_t>(num_points));
                break;
        }
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
    Vector3f origin;
    Vector3f direction;
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

    py::class_<Vector3f>(m, "vec3", "A 3D vector with float components.")
        .def(py::init<float, float, float>(), "x"_a=0.f, "y"_a=0.f, "z"_a=0.f)
        .def_readwrite("x", &Vector3f::x, "X-component of the vector.")
        .def_readwrite("y", &Vector3f::y, "Y-component of the vector.")
        .def_readwrite("z", &Vector3f::z, "Z-component of the vector.")
        .def("__repr__", [](const Vector3f &v) {
            return "vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    py::class_<PyRay>(m, "Ray", "Represents a ray for intersection queries.")
        .def(py::init<const Vector3f&, const Vector3f&, float>(),
             "origin"_a, "direction"_a, "t"_a = 1e30f)
        .def_readwrite("origin", &PyRay::origin, "The origin point of the ray.")
        .def_readwrite("direction", &PyRay::direction, "The direction vector of the ray.")
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


    py::class_<BVHWrapper>(m, "BVH", "A Bounding Volume Hierarchy for fast ray intersections.")
        .def_static("from_triangles", &BVHWrapper::from_triangles,
            R"((
                Builds a BVH for triangles from a (N, 3, 3) or (N, 9) float array.

                Args:
                    triangles (numpy.ndarray): A float32 array of shape (N, 3, 3) or (N, 9).
                    quality (BuildQuality): The desired build quality.

                Returns:
                    BVH: A new BVH instance.
            ))",
            py::arg("triangles").noconvert(), py::arg("quality") = BuildQuality::Balanced)

        .def_static("from_points", &BVHWrapper::from_points,
             R"((
                Builds a BVH for points from a (N, 3) float array.

                Internally, this represents each point as a small axis-aligned bounding box.

                Args:
                    points (numpy.ndarray): A float32 array of shape (N, 3) representing N points.
                    radius (float): The radius used to create an AABB for each point.
                    quality (BuildQuality): The desired build quality.

                Returns:
                    BVH: A new BVH instance.
            ))",
            py::arg("points").noconvert(), py::arg("radius") = 1e-5f, py::arg("quality") = BuildQuality::Balanced)

        .def_static("from_vertices_4f", &BVHWrapper::from_vertices_4f,
            R"((
                Builds a BVH from a pre-formatted (M, 4) vertex array.

                This is a zero-copy operation. The BVH will hold a reference to the
                provided numpy array's memory buffer. The array must not be garbage-collected
                while the BVH is in use. M must be a multiple of 3.

                Args:
                    vertices (numpy.ndarray): A float32 array of shape (M, 4).
                    quality (BuildQuality): The desired build quality.

                Returns:
                    BVH: A new BVH instance.
            ))",
            py::arg("vertices").noconvert(), py::arg("quality") = BuildQuality::Balanced)

        .def("save", [](const BVHWrapper& self, const py::object& filepath_obj) {
            std::string filepath = py::str(filepath_obj).cast<std::string>();
            self.bvh->Save(filepath.c_str());
        }, py::arg("filepath"),
            R"((
                Saves the BVH to a file.

                Args:
                    filepath (str or pathlib.Path): The path where the BVH file will be saved.
            ))"
        )

        .def_static("load", [](py::array_t<float, py::array::c_style | py::array::forcecast> vertices_np, const py::object& filepath_obj) {
            if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
                throw std::runtime_error("Vertex data for loading must be a 2D numpy array with shape (M, 4).");
            }

            std::string filepath = py::str(filepath_obj).cast<std::string>();

            auto wrapper = std::make_unique<BVHWrapper>();
            wrapper->source_geometry = vertices_np; // Keep ref to vertices alive

            py::buffer_info buf = vertices_np.request();
            auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(buf.ptr);
            uint32_t prim_count = buf.shape[0] / 3;

            bool success = wrapper->bvh->Load(filepath.c_str(), vertices_ptr, prim_count);
            if (!success) {
                throw std::runtime_error("Failed to load BVH from file. Check file integrity, version, and geometry compatibility.");
            }

            // Infer quality from refittable flag
            wrapper->quality = wrapper->bvh->refittable ? BuildQuality::Balanced : BuildQuality::High;

            return wrapper;

        }, py::arg("vertices").noconvert(), py::arg("filepath"),
            R"((
                Loads a BVH from a file, requires the original vertex data.

                Per design tinybvh does not save the geometric vertex data in its file format;
                only the acceleration structure itself (the nodes and primitive indices).

                Thus, the vertex data that was used to create the original BVH must be provided:
                this function re-links the loaded acceleration structure to the vertex data in memory.

                Args:
                    vertices (numpy.ndarray): A float32, C-style contiguous numpy array of shape (M, 4) representing
                        the vertex data. This must be the same data that was used when the BVH was originally built and saved.
                    filepath (str or pathlib.Path): The path to the saved BVH file.

                Returns:
                    BVH: A new BVH instance.
            ))")

        .def_property_readonly("nodes", [](BVHWrapper &self) -> py::array {
            if (!self.bvh || self.bvh->usedNodes == 0) {
                py::dtype dt = py::dtype::of<tinybvh::BVH::BVHNode>();
                return py::array(dt, {0}, {});
            }
            return py::array_t<tinybvh::BVH::BVHNode>(
                { (py::ssize_t)self.bvh->usedNodes }, self.bvh->bvhNode, py::cast(self)
            );
        }, "The structured numpy array of BVH nodes.")

        .def_property_readonly("prim_indices", [](BVHWrapper &self) -> py::array {
            if (!self.bvh || self.bvh->idxCount == 0) {
                return py::array_t<uint32_t>(0);
            }
            return py::array_t<uint32_t>(
                { (py::ssize_t)self.bvh->idxCount }, // shape
                self.bvh->primIdx,    // data pointer
                py::cast(self)        // owner
            );
        }, "The array of primitive indices, ordered for locality.")

        .def_property_readonly("quality", [](const BVHWrapper &self) { return self.quality; },
            "The build quality level used to construct the BVH.")

        .def("intersect", [](BVHWrapper &self, PyRay &py_ray) -> float {
            tinybvh::Ray ray(
                tinybvh::bvhvec3(py_ray.origin.x, py_ray.origin.y, py_ray.origin.z),
                tinybvh::bvhvec3(py_ray.direction.x, py_ray.direction.y, py_ray.direction.z),
                py_ray.t
            );

            self.bvh->Intersect(ray);

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

        .def("intersect_batch", [](BVHWrapper &self, py::array_t<float, py::array::c_style> origins_np,
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

        .def("is_occluded", [](BVHWrapper &self, const PyRay &py_ray) -> bool {
            tinybvh::Ray ray(
                tinybvh::bvhvec3(py_ray.origin.x, py_ray.origin.y, py_ray.origin.z),
                tinybvh::bvhvec3(py_ray.direction.x, py_ray.direction.y, py_ray.direction.z),
                py_ray.t // The ray's t is used as the maximum distance for the occlusion check
            );
            return self.bvh->IsOccluded(ray);
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

        .def("is_occluded_batch", [](BVHWrapper &self, py::array_t<float, py::array::c_style> origins_np,
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

        .def("refit", &BVHWrapper::refit,
             R"((
                Refits the BVH to the current state of the source geometry, which is much faster than a full rebuild.

                Should be called after the underlying vertex data (numpy array used for construction)
                has been modified.

                Note: This will fail if the BVH was built with spatial splits (with BuildQuality.High).
             ))")

        .def_property_readonly("node_count", [](const BVHWrapper &self){ return self.bvh->usedNodes; }, "Total number of nodes in the BVH.")
        .def_property_readonly("prim_count", [](const BVHWrapper &self){ return self.bvh->triCount; }, "Total number of primitives in the BVH.")
        .def_property_readonly("aabb_min", [](const BVHWrapper &self){
            return Vector3f(self.bvh->aabbMin.x, self.bvh->aabbMin.y, self.bvh->aabbMin.z);
        }, "The minimum corner of the root axis-aligned bounding box.")
        .def_property_readonly("aabb_max", [](const BVHWrapper &self){
            return Vector3f(self.bvh->aabbMax.x, self.bvh->aabbMax.y, self.bvh->aabbMax.z);
        }, "The maximum corner of the root axis-aligned bounding box.");
}