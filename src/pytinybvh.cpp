#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define GLM_ENABLE_EXPERIMENTAL
#pragma warning(disable : 4201)
#include <glm/glm/glm.hpp>

#define TINYBVH_IMPLEMENTATION
#include "tinybvh/tiny_bvh.h"

#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;  // for '_a' literals


py::tuple build_bvh_generic(tinybvh::BVH& bvh, const std::vector<tinybvh::bvhvec4>& vertices, size_t prim_count) {
    // the build function takes a pointer to the start of the vertex data, and the number of primitives

    bvh.Build(vertices.data(), static_cast<uint32_t>(prim_count));

    // Copy the results into numpy arrays
    // (member names are bvhNode, primIdx, and usedNodes)
    std::vector<py::ssize_t> node_shape = {(py::ssize_t)bvh.usedNodes, 8};
    py::array_t<float> bvh_nodes_np(node_shape);

    memcpy(
        bvh_nodes_np.mutable_data(),
        bvh.bvhNode,
        bvh.usedNodes * sizeof(tinybvh::BVH::BVHNode)
    );

    py::array_t<uint32_t> prim_indices_np((py::ssize_t)bvh.idxCount);
    memcpy(
        prim_indices_np.mutable_data(),
        bvh.primIdx,
        bvh.idxCount * sizeof(uint32_t)
    );

    return py::make_tuple(bvh_nodes_np, prim_indices_np);
}

// Python-exposed function for triangles
py::tuple from_triangles(py::array_t<float, py::array::c_style | py::array::forcecast> tris_np) {
    if (tris_np.ndim() != 2 || tris_np.shape(1) != 9) {
        throw std::runtime_error("Input must be a 2D numpy array with shape (N, 9).");
    }

    size_t num_tris = tris_np.shape(0);
    float* ptr = static_cast<float*>(tris_np.request().ptr);

    std::vector<tinybvh::bvhvec4> vertices;
    vertices.reserve(num_tris * 3);
    for (size_t i = 0; i < num_tris; ++i) {
        vertices.emplace_back(ptr[i * 9 + 0], ptr[i * 9 + 1], ptr[i * 9 + 2], 0.0f);
        vertices.emplace_back(ptr[i * 9 + 3], ptr[i * 9 + 4], ptr[i * 9 + 5], 0.0f);
        vertices.emplace_back(ptr[i * 9 + 6], ptr[i * 9 + 7], ptr[i * 9 + 8], 0.0f);
    }

    tinybvh::BVH bvh;
    return build_bvh_generic(bvh, vertices, num_tris);
}

// Python-exposed function for points
py::tuple from_points(py::array_t<float, py::array::c_style | py::array::forcecast> points_np, float radius) {
    if (points_np.ndim() != 2 || points_np.shape(1) != 3) {
        throw std::runtime_error("Input must be a 2D numpy array with shape (N, 3).");
    }

    if (radius < 0.0f) {
        throw std::runtime_error("Point radius cannot be negative.");
    }

    size_t num_points = points_np.shape(0);
    float* ptr = static_cast<float*>(points_np.request().ptr);

    // Create (almost) degenerate triangles for each point
    std::vector<tinybvh::bvhvec4> vertices;
    vertices.reserve(num_points * 3);
    for (size_t i = 0; i < num_points; ++i) {
        float x = ptr[i * 3 + 0];
        float y = ptr[i * 3 + 1];
        float z = ptr[i * 3 + 2];
        vertices.emplace_back(x - radius, y - radius, z - radius, 0.0f); // Min corner
        vertices.emplace_back(x + radius, y + radius, z + radius, 0.0f); // Max corner
        vertices.emplace_back(x - radius, y - radius, z - radius, 0.0f); // Repeat min corner to form a "triangle"
    }

    tinybvh::BVH bvh;
    return build_bvh_generic(bvh, vertices, num_points);
}

// Python module definition
PYBIND11_MODULE(pytinybvh, m) {
    m.doc() = "Python bindings for the tinybvh library";
    m.def("from_triangles", &from_triangles, "Builds a BVH for triangles from a (N, 9) float array.");
    m.def("from_points", &from_points,
          "Builds a BVH for points from a (N, 3) float array.",
          "points"_a, "radius"_a = 1e-5f); // tiny default radius for numerical stability
}