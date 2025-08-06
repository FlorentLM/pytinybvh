#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define TINYBVH_IMPLEMENTATION
#include "tinybvh/tiny_bvh.h"

#include <vector>
#include <stdexcept>
#include <memory> // for std::unique_ptr
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

// mini 3D vector struct to avoid having a dependency on glm just for 3 floats (might change in the future tho)
struct Vector3f {
    float x = 0.f, y = 0.f, z = 0.f;

    Vector3f() = default;
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
};

// C++ class to wrap the tinybvh::BVH object and its associated data: this is the object held by the Python BVH class
struct BVHWrapper {
    std::unique_ptr<tinybvh::BVH> bvh;

    py::array source_geometry; // ref to the Python numpy array that contains the source vertices
    // This prevents Python's garbage collector from freeing the memory while the C++ BVH object might still be using it
    // (that's important especially for refitting)

    BVHWrapper() : bvh(std::make_unique<tinybvh::BVH>()) {}

    static std::unique_ptr<BVHWrapper> from_vertices_4f(py::array_t<float, py::array::c_style | py::array::forcecast> vertices_np) {
        if (vertices_np.ndim() != 2 || vertices_np.shape(1) != 4) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (M, 4).");
        }
        if (vertices_np.shape(0) % 3 != 0) {
            throw std::runtime_error("Input vertex count must be a multiple of 3.");
        }

        auto wrapper = std::make_unique<BVHWrapper>();
        // no copy: ref to the numpy array
        wrapper->source_geometry = vertices_np;

        py::buffer_info buf = vertices_np.request();
        auto vertices_ptr = static_cast<const tinybvh::bvhvec4*>(buf.ptr);
        size_t prim_count = buf.shape[0] / 3;

        // no copy: build directly from the numpy array's memory buffer
        wrapper->bvh->Build(vertices_ptr, static_cast<uint32_t>(prim_count));

        return wrapper;
    }

    static std::unique_ptr<BVHWrapper> from_triangles(py::array_t<float, py::array::c_style | py::array::forcecast> tris_np) {
        bool shape_ok = (tris_np.ndim() == 2 && tris_np.shape(1) == 9) ||
                        (tris_np.ndim() == 3 && tris_np.shape(1) == 3 && tris_np.shape(2) == 3);
        if (!shape_ok) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9) or a 3D array with shape (N, 3, 3).");
        }

        auto wrapper = std::make_unique<BVHWrapper>();

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
        wrapper->bvh->Build(vertices_ptr, static_cast<uint32_t>(num_tris));

        return wrapper;
    }

    static std::unique_ptr<BVHWrapper> from_points(py::array_t<float, py::array::c_style | py::array::forcecast> points_np, float radius) {
        if (points_np.ndim() != 2 || points_np.shape(1) != 3) {
            throw std::runtime_error("Input must be a 2D numpy array with shape (N, 3).");
        }
        if (radius <= 0.0f) {
            throw std::runtime_error("Point radius must be positive.");
        }

        auto wrapper = std::make_unique<BVHWrapper>();

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

        wrapper->bvh->Build(vertices_ptr, static_cast<uint32_t>(num_points));

        return wrapper;
    }

    void refit() {
        if (!bvh || !bvh->refittable) {
            throw std::runtime_error("BVH is not refittable (it might have been built with spatial splits).");
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

    PYBIND11_NUMPY_DTYPE(tinybvh::bvhvec3, x, y, z);
    PYBIND11_NUMPY_DTYPE_EX(tinybvh::BVH::BVHNode, aabbMin, "aabb_min", leftFirst, "left_first", aabbMax, "aabb_max", triCount, "prim_count");

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
            "Builds a BVH for triangles from a (N, 3, 3) or (N, 9) float array.",
            py::arg("triangles").noconvert("Must be a numpy.ndarray"))
        .def_static("from_points", &BVHWrapper::from_points,
            "Builds a BVH for points from a (N, 3) float array.",
            py::arg("points").noconvert("Must be a numpy.ndarray"), py::arg("radius") = 1e-5f)
        .def_static("from_vertices_4f", &BVHWrapper::from_vertices_4f,
            "Builds a BVH from a pre-formatted (N*3, 4) vertex array. This is a zero-copy operation.",
            py::arg("vertices").noconvert("Must be a numpy.ndarray"))

        // Expose internal BVH nodes array as zero-copy array view
        .def_property_readonly("nodes", [](BVHWrapper &self) -> py::array {
            if (!self.bvh || self.bvh->usedNodes == 0) {
                // empty array with the correct structured dtype
                py::dtype dt = py::dtype::of<tinybvh::BVH::BVHNode>();
                return py::array(dt, {0}, {});
            }
            // create 1D array of BVHNode structs, Pybind11 will automatically use the structured dtype
            return py::array_t<tinybvh::BVH::BVHNode>(
                { (py::ssize_t)self.bvh->usedNodes }, // shape (N,)
                self.bvh->bvhNode,                   // data pointer
                py::cast(self)                      // This py::cast tells numpy that BVHWrapper owns this memory
            );
        }, "The structured numpy array of BVH nodes.")

        // Expose primitive index array as zero-copy array view
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

        .def("intersect", [](BVHWrapper &self, PyRay &py_ray) -> bool {

            // python Ray -> C++ Ray
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
                return true;
            }
            return false;
        }, py::arg("ray"), "Performs an intersection query with a ray. Modifies the ray object in-place.")

        .def("refit", &BVHWrapper::refit, "Refits the BVH to the current state of the source geometry.")

        .def_property_readonly("node_count", [](const BVHWrapper &self){ return self.bvh->usedNodes; }, "Total number of nodes in the BVH.")
        .def_property_readonly("prim_count", [](const BVHWrapper &self){ return self.bvh->triCount; }, "Total number of primitives in the BVH.")
        .def_property_readonly("aabb_min", [](const BVHWrapper &self){
            return Vector3f(self.bvh->aabbMin.x, self.bvh->aabbMin.y, self.bvh->aabbMin.z);
        }, "The minimum corner of the root axis-aligned bounding box.")
        .def_property_readonly("aabb_max", [](const BVHWrapper &self){
            return Vector3f(self.bvh->aabbMax.x, self.bvh->aabbMax.y, self.bvh->aabbMax.z);
        }, "The maximum corner of the root axis-aligned bounding box.");
}