import time
from pathlib import Path
import numpy as np
import pytest
from typing import Union
from pytinybvh import BVH, Ray, BuildQuality
import trimesh


# ==============================================================================
# HELPERS AND FIXTURES
# ==============================================================================

def translation_matrix(translation: np.ndarray) -> np.ndarray:
    M = np.identity(4, dtype=np.float32)
    M[:3, 3] = translation
    return M


def scale_matrix(scale: Union[float, np.ndarray]) -> np.ndarray:
    M = np.identity(4, dtype=np.float32)
    np.fill_diagonal(M, (*([scale] * 3), 1.0))
    return M


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1 - c
    xs, ys, zs = x * s, y * s, z * s
    xC, yC, zC = x * C, y * C, z * C
    xyC, yzC, zxC = x * yC, y * zC, z * xC
    return np.array([
        [x * xC + c, xyC - zs, zxC + ys, 0],
        [xyC + zs, y * yC + c, yzC - xs, 0],
        [zxC - ys, yzC + xs, z * zC + c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


@pytest.fixture(scope="module")
def bvh_two_triangles():
    """Fixture for a simple BVH with two triangles"""

    triangles = np.array([
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],  # Tri 0 at z=0
        [[2.0, 2.0, 5.0], [4.0, 2.0, 5.0], [3.0, 4.0, 5.0]],  # Tri 1 at z=5
    ], dtype=np.float32)
    bvh = BVH.from_triangles(triangles, quality=BuildQuality.Balanced)
    return bvh, triangles

@pytest.fixture
def bvh_cube():
    """Fixture for a simple BVH of a unit cube"""

    cube_verts = np.zeros((8, 4), dtype=np.float32)
    cube_verts[:, :3] = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    cube_indices = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 7], [0, 7, 3],
        [1, 5, 6], [1, 6, 2], [0, 1, 5], [0, 5, 4], [3, 2, 6], [3, 6, 7]
    ], dtype=np.uint32)

    bvh = BVH.from_indexed_mesh(cube_verts, cube_indices, quality=BuildQuality.Balanced)

    return bvh, cube_verts, cube_indices

@pytest.fixture
def bvh_from_ply():
    """Fixture that loads a complex mesh from a PLY file"""
    ply_path = Path("assets/sneks.ply")

    mesh = trimesh.load(ply_path, process=False)
    verts_3d = np.array(mesh.vertices, dtype=np.float32)
    indices = np.array(mesh.faces, dtype=np.uint32)

    # Center and normalize
    verts_3d -= np.mean(verts_3d, axis=0)
    verts_3d /= np.max(np.abs(verts_3d))

    verts_4d = np.zeros((verts_3d.shape[0], 4), dtype=np.float32)
    verts_4d[:, :3] = verts_3d

    bvh = BVH.from_indexed_mesh(verts_4d, indices, quality=BuildQuality.Balanced)
    return bvh

@pytest.fixture(scope="module")
def tlas_scene():
    """Fixture for a complete TLAS scene with two BLASes and four instances"""

    # BLAS 0: Unit cube
    cube_verts = np.zeros((8, 4), dtype=np.float32)
    cube_verts[:, :3] = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    cube_indices = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 7], [0, 7, 3],
        [1, 5, 6], [1, 6, 2], [0, 1, 5], [0, 5, 4], [3, 2, 6], [3, 6, 7]
    ], dtype=np.uint32)
    bvh_cube_blas = BVH.from_indexed_mesh(cube_verts, cube_indices)

    # BLAS 1: A 2x2 quad on the XY plane
    quad_verts = np.zeros((4, 4), dtype=np.float32)
    quad_verts[:, :3] = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    quad_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    bvh_quad_blas = BVH.from_indexed_mesh(quad_verts, quad_indices)

    blases = [bvh_cube_blas, bvh_quad_blas]

    # Define Instances
    instance_dtype = np.dtype([('transform', '<f4', (4, 4)), ('blas_id', '<u4'), ('mask', '<u4')])
    instances = np.zeros(4, dtype=instance_dtype)
    instances[0] = (np.identity(4, dtype=np.float32), 0, 0b0001)
    instances[1] = (translation_matrix(np.array([5, 0, 0])) @ scale_matrix(2.0), 0, 0b0010)
    instances[2] = (translation_matrix(np.array([0, 5, 0])) @ rotation_matrix(np.array([0, 1, 0]), np.pi / 2), 1,
                    0b0100)
    instances[3] = (translation_matrix(np.array([0, 0, -5])), 1, 0b1000)

    tlas_bvh = BVH.build_tlas(instances, blases)

    return {
        "tlas": tlas_bvh,
        "blases": blases,
        "instances": instances,
        "cube_verts": cube_verts,
        "cube_indices": cube_indices,
        "quad_verts": quad_verts,
        "quad_indices": quad_indices,
    }


# ==============================================================================
# TEST CLASSES
# ==============================================================================

class TestConstruction:
    def test_from_triangles_quality(self, bvh_two_triangles):
        """Tests that BVHs can be built with different quality settings"""

        bvh, triangles = bvh_two_triangles
        assert bvh.quality == BuildQuality.Balanced

        bvh_high = BVH.from_triangles(triangles, quality=BuildQuality.High)
        bvh_quick = BVH.from_triangles(triangles, quality=BuildQuality.Quick)

        assert bvh_high.quality == BuildQuality.High
        assert bvh_quick.quality == BuildQuality.Quick

    def test_from_vertices(self, bvh_two_triangles):
        """Tests the zero-copy `from_vertices` builder"""

        _, triangles = bvh_two_triangles
        vertices_4d = np.zeros((6, 4), dtype=np.float32)
        vertices_4d[:, :3] = triangles.reshape(6, 3)
        bvh = BVH.from_vertices(vertices_4d)
        assert bvh.prim_count == 2
        ray = Ray(origin=(0, 0, -1), direction=(0, 0, 1))
        bvh.intersect(ray)
        assert np.isclose(ray.t, 1.0)

    def test_from_indexed_mesh(self):
        """Tests the zero-copy `from_indexed_mesh` builder"""

        verts_3d = np.array([[0, 0, 10], [5, 0, 10], [0, 5, 10], [5, 5, 10]], dtype=np.float32)
        verts_4d = np.zeros((4, 4), dtype=np.float32)
        verts_4d[:, :3] = verts_3d
        indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        bvh = BVH.from_indexed_mesh(verts_4d, indices)
        assert bvh.prim_count == 2
        ray = Ray(origin=(2.5, 2.5, 0), direction=(0, 0, 1))
        bvh.intersect(ray)
        assert np.isclose(ray.t, 10.0)


class TestCoreFunctionality:
    def test_properties(self, bvh_two_triangles):
        """Tests basic properties like node_count, prim_count, and aabbs"""

        bvh, _ = bvh_two_triangles
        assert bvh.prim_count == 2
        assert bvh.node_count > 0  # exact number depends on build
        assert bvh.nodes.shape == (bvh.node_count,)
        assert bvh.prim_indices.shape == (bvh.prim_count,)
        assert bvh.aabb_min.shape == (3,)
        assert bvh.aabb_max.shape == (3,)
        assert np.all(bvh.aabb_min <= bvh.aabb_max)

    def test_save_load(self, tmp_path):
        """Tests saving and loading for both indexed and non-indexed BVHs"""

        # Non-indexed
        verts_4d = np.zeros((6, 4), dtype=np.float32)
        bvh_soup = BVH.from_vertices(verts_4d)
        filepath_soup = tmp_path / "soup.bvh"
        bvh_soup.save(filepath_soup)
        bvh_loaded_soup = BVH.load(filepath_soup, verts_4d)
        np.testing.assert_array_equal(bvh_loaded_soup.nodes, bvh_soup.nodes)

        # Indexed
        indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        bvh_indexed = BVH.from_indexed_mesh(verts_4d[:4], indices)
        filepath_indexed = tmp_path / "indexed.bvh"
        bvh_indexed.save(filepath_indexed)
        bvh_loaded_indexed = BVH.load(filepath_indexed, verts_4d[:4], indices)
        np.testing.assert_array_equal(bvh_loaded_indexed.nodes, bvh_indexed.nodes)

    def test_refit(self):
        """Tests that refitting the BVH correctly updates to new vertex positions"""

        verts_4d = np.zeros((4, 4), dtype=np.float32)
        verts_4d[:, :3] = np.array([[0, 0, 10], [5, 0, 10], [0, 5, 10], [5, 5, 10]])
        indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        bvh = BVH.from_indexed_mesh(verts_4d, indices)

        # Modify vertices in-place and refit
        verts_4d[:, 2] = 20.0
        bvh.refit()

        ray = Ray(origin=(2.5, 2.5, 0), direction=(0, 0, 1))
        hit_dist = bvh.intersect(ray)
        assert np.isclose(hit_dist, 20.0)

        # Refitting a high-quality BVH should fail
        bvh_high = BVH.from_indexed_mesh(verts_4d, indices, quality=BuildQuality.High)
        with pytest.raises(RuntimeError):
            bvh_high.refit()


class TestIntersection:
    def test_single_intersect_hit_miss(self, bvh_two_triangles):
        """Tests single ray intersections for both hits and misses"""

        bvh, _ = bvh_two_triangles

        # Hit
        ray_hit = Ray(origin=(0, 0, -10), direction=(0, 0, 1))
        hit_dist = bvh.intersect(ray_hit)
        assert np.isclose(hit_dist, 10.0)
        assert ray_hit.prim_id == 0 and np.isclose(ray_hit.t, 10.0)

        # Miss
        ray_miss = Ray(origin=(10, 10, -10), direction=(0, 0, 1))
        miss_dist = bvh.intersect(ray_miss)
        assert np.isinf(miss_dist)
        assert ray_miss.prim_id == np.iinfo(np.uint32).max

    def test_barycentric_coords(self, bvh_two_triangles):
        """Tests that barycentric coordinates are computed correctly"""
        bvh, _ = bvh_two_triangles
        # Ray hitting the center of the first triangle's base edge
        ray = Ray(origin=(0, -1, -1), direction=(0, 0, 1))
        bvh.intersect(ray)
        assert np.isclose(ray.u, 0.5) and np.isclose(ray.v, 0.0)

    def test_intersect_batch(self, bvh_two_triangles):
        """Tests batch intersection with hits, misses, and t_max"""

        bvh, _ = bvh_two_triangles
        origins = np.array([
            [0.0, 0.0, -10.0],  # Ray 0: Hit tri 0 (z=0) at t=10
            [3.0, 3.0, -10.0],  # Ray 1: Hit tri 1 (z=5) at t=15
            [10.0, 10.0, -10.0],  # Ray 2: Miss
            [0.0, 0.0, -10.0],  # Ray 3: Aimed at tri 0, but t_max is too short
        ], dtype=np.float32)
        directions = np.array([[0, 0, 1]] * 4, dtype=np.float32)
        t_max = np.array([100.0, 100.0, 100.0, 5.0], dtype=np.float32)

        hits = bvh.intersect_batch(origins, directions, t_max)
        prim_ids = hits['prim_id'].astype(np.int32)

        assert prim_ids[0] == 0 and np.isclose(hits[0]['t'], 10.0)
        assert prim_ids[1] == 1 and np.isclose(hits[1]['t'], 15.0)
        assert prim_ids[2] == -1 and np.isinf(hits[2]['t'])
        assert prim_ids[3] == -1 and np.isinf(hits[3]['t'])


class TestOcclusion:
    def test_single_is_occluded(self, bvh_two_triangles):
        """Tests single ray occlusion queries"""

        bvh, _ = bvh_two_triangles
        # Occluded: ray hits tri 1 at t=15, max_t=100
        ray_occluded = Ray(origin=(3, 3, -10), direction=(0, 0, 1), t=100.0)
        assert bvh.is_occluded(ray_occluded)

        # Not occluded: ray misses
        ray_miss = Ray(origin=(10, 10, -10), direction=(0, 0, 1))
        assert not bvh.is_occluded(ray_miss)

        # Not occluded: ray aimed at tri 1 (t=15), but max_t is 10
        ray_t_limited = Ray(origin=(3, 3, -10), direction=(0, 0, 1), t=10.0)
        assert not bvh.is_occluded(ray_t_limited)

    def test_is_occluded_batch(self, bvh_two_triangles):
        """Tests batch occlusion queries"""

        bvh, _ = bvh_two_triangles

        origins = np.array([
            [0.0, 0.0, -10.0],  # Ray 0: Occluded by tri 0 (t=10)
            [3.0, 3.0, -10.0],  # Ray 1: Not occluded by tri 1 (t=15) due to t_max
            [10.0, 10.0, -10.0],  # Ray 2: Not occluded (miss)
        ], dtype=np.float32)

        directions = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        t_max = np.array([100.0, 10.0, 100.0], dtype=np.float32)

        occlusion = bvh.is_occluded_batch(origins, directions, t_max)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(occlusion, expected)


class TestTLAS:
    def test_tlas_creation(self, tlas_scene):
        """Tests that the TLAS is created with the correct number of instances"""

        assert tlas_scene["tlas"].prim_count == 4

    def test_tlas_intersections(self, tlas_scene):
        """Tests ray intersections with different instances in the TLAS"""

        tlas = tlas_scene["tlas"]

        # Hit instance 0: Unit cube at origin
        ray0 = Ray(origin=(0, 0, -2), direction=(0, 0, 1))
        tlas.intersect(ray0)
        assert np.isclose(ray0.t, 1.5) and ray0.inst_id == 0 and ray0.prim_id in [0, 1]

        # Hit instance 1: Scaled cube at (5, 0, 0)
        ray1 = Ray(origin=(5, 0, -2), direction=(0, 0, 1))
        tlas.intersect(ray1)
        assert np.isclose(ray1.t, 1.0) and ray1.inst_id == 1

        # Hit instance 2: Rotated quad at (0, 5, 0)
        ray2 = Ray(origin=(-2, 5, 0), direction=(1, 0, 0))
        tlas.intersect(ray2)
        assert np.isclose(ray2.t, 2.0) and ray2.inst_id == 2

    def test_tlas_masking(self, tlas_scene):
        """Tests that ray masks correctly filter instances"""
        tlas = tlas_scene["tlas"]

        # Ray starts in front of instance 0, aimed toward instance 3.
        # Mask 0b1000 should ignore instance 0 and hit instance 3.
        ray = Ray(origin=(0, 0, 2), direction=(0, 0, -1), mask=0b1000)

        tlas.intersect(ray)

        assert np.isclose(ray.t, 7.0) and ray.inst_id == 3


class TestPostProcessing:
    def test_optimize_improves_sah_score(self, bvh_from_ply):
        """
        Tests that optimizing a complex BVH reduces its SAH cost.
        Note: BVH built with BuildQuality.Quick because it produces a
        less optimal tree that has more room for improvement.
        """
        bvh = bvh_from_ply  # This BVH is from a complex mesh

        # Get SAH cost before optimization
        sah_before = bvh.sah_cost
        assert sah_before > 0.0 and np.isfinite(sah_before)

        # Optimize the BVH
        bvh.optimize()

        # Get SAH cost after optimization
        sah_after = bvh.sah_cost
        assert sah_after > 0.0 and np.isfinite(sah_after)

        print(f"\nSAH Cost before optimization: {sah_before:.4f}")
        print(f"SAH Cost after optimization:  {sah_after:.4f}")

        # The core assertion: SAH cost should decrease.
        assert sah_after < sah_before

        # Test that the BVH is still refittable after optimization
        # (tinybvh's reinsertion optimization does not introduce spatial splits)
        try:
            bvh.refit()
        except RuntimeError:
            pytest.fail("BVH should still be refittable after optimization.")

    def test_optimize_improves_performance(self, bvh_from_ply):
        """
        Tests that optimizing a BVH improves real-world query performance.
        This is the most reliable test, as the library's SAHCost metric can
        be misleading.
        """
        bvh = bvh_from_ply  # Fixture now correctly uses BuildQuality.Balanced

        # Generate a large number of random rays aimed at the BVH's AABB
        num_rays = 100_000
        aabb_min, aabb_max = bvh.aabb_min, bvh.aabb_max
        aabb_center = (aabb_min + aabb_max) / 2.0
        aabb_size = float(np.max(aabb_max - aabb_min))

        # Generate random origins on a sphere outside the AABB
        phi = np.random.uniform(0, np.pi, num_rays)
        theta = np.random.uniform(0, 2 * np.pi, num_rays)
        origins = np.zeros((num_rays, 3), dtype=np.float32)
        origins[:, 0] = aabb_center[0] + aabb_size * np.sin(phi) * np.cos(theta)
        origins[:, 1] = aabb_center[1] + aabb_size * np.sin(phi) * np.sin(theta)
        origins[:, 2] = aabb_center[2] + aabb_size * np.cos(phi)

        # Aim rays towards the AABB center
        directions = aabb_center - origins
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        # Time intersection before optimization
        start_time_before = time.perf_counter()
        hits_before = bvh.intersect_batch(origins, directions)
        end_time_before = time.perf_counter()
        time_before = end_time_before - start_time_before

        # Optimize the BVH
        bvh.optimize()

        # Time intersection after optimization
        start_time_after = time.perf_counter()
        hits_after = bvh.intersect_batch(origins, directions)
        end_time_after = time.perf_counter()
        time_after = end_time_after - start_time_after

        print(f"\nIntersection time before optimization: {time_before:.6f}s")
        print(f"Intersection time after optimization:  {time_after:.6f}s")

        # Verify correctness before checking performance
        # Primitive and instance IDs must be identical
        np.testing.assert_array_equal(hits_before['prim_id'], hits_after['prim_id'])
        np.testing.assert_array_equal(hits_before['inst_id'], hits_after['inst_id'])

        # mask for rays that hit in both cases
        hit_mask = (hits_before['t'] != np.inf) & (hits_after['t'] != np.inf)

        # Compare hit distances only for the hits, allowing for float tolerance
        np.testing.assert_allclose(
            hits_before['t'][hit_mask],
            hits_after['t'][hit_mask],
            rtol=1e-5
        )

        # The core assertion: performance should improve
        assert time_after < time_before

    def test_optimize_maintains_correctness(self, bvh_cube):
        """Tests that the BVH produces identical intersection results after optimization"""

        bvh, _, _ = bvh_cube

        # Define a set of rays to test against the cube
        origins = np.array([
            [0, 0, -2],  # Hit -Z face
            [0, 0, 2],   # Hit +Z face
            [-2, 0, 0],  # Hit -X face
            [2, 0, 0],   # Hit +X face
            [0, -2, 0],  # Hit -Y face
            [0, 2, 0],   # Hit +Y face
            [5, 5, 5],   # Miss
        ], dtype=np.float32)
        directions = np.array([
            [0, 0, 1],
            [0, 0, -1],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [-1, -1, -1],
        ], dtype=np.float32)

        # Get results before optimization
        hits_before = bvh.intersect_batch(origins, directions)

        # Optimize
        bvh.optimize()
        hits_after = bvh.intersect_batch(origins, directions)

        # Compare results
        # Primitive and instance IDs must be identical
        np.testing.assert_array_equal(hits_before['prim_id'], hits_after['prim_id'])
        np.testing.assert_array_equal(hits_before['inst_id'], hits_after['inst_id'])

        # mask for rays that hit in both cases
        hit_mask = (hits_before['t'] != np.inf) & (hits_after['t'] != np.inf)

        # Compare hit distances only for the hits, allowing for float tolerance
        np.testing.assert_allclose(
            hits_before['t'][hit_mask],
            hits_after['t'][hit_mask],
            rtol=1e-5
        )

    def test_compact_on_compact_bvh(self, bvh_cube):
        """
        Tests that compact() on an already compact BVH is a no-op and doesn't
        corrupt the data

        A better test would require a build process
        that is *guaranteed* to create a non-compact tree (a threaded build ?)
        """
        bvh, _, _ = bvh_cube
        initial_nodes = bvh.nodes.copy()
        initial_node_count = bvh.node_count

        # Compacting an already compact BVH should not change it
        bvh.compact()

        assert bvh.node_count == initial_node_count
        np.testing.assert_array_equal(bvh.nodes, initial_nodes)

        # Also verify it still works
        ray = Ray(origin=(0, 0, -2), direction=(0, 0, 1))
        bvh.intersect(ray)
        assert np.isclose(ray.t, 1.5)


class TestRobustness:
    def test_empty_inputs(self):
        """Tests that the library handles empty geometry and ray batches gracefully"""

        bvh = BVH.from_triangles(np.empty((0, 3, 3), dtype=np.float32))

        assert bvh.prim_count == 0 and bvh.node_count == 0

        ray = Ray((0, 0, 0), (0, 0, 1))

        assert np.isinf(bvh.intersect(ray))

        hits = bvh.intersect_batch(np.empty((0, 3)), np.empty((0, 3)))

        assert len(hits) == 0

    def test_invalid_shapes_and_types(self):
        """Tests that builders raise appropriate errors for invalid input shapes and dtypes"""

        # Convenience builders should raise RuntimeError for bad shapes
        with pytest.raises(RuntimeError): BVH.from_triangles(np.zeros((5, 8)))
        with pytest.raises(RuntimeError): BVH.from_points(np.zeros((5, 4)))

        # Core builders expect specific dtypes and will fail with TypeError from pybind11
        with pytest.raises(TypeError): BVH.from_vertices(np.zeros((6, 4)))  # float64
        with pytest.raises(TypeError):
            BVH.from_indexed_mesh(np.zeros((4, 4)), np.zeros((2, 3), dtype=np.uint32))  # float64

        # Core builders should raise RuntimeError for bad shapes if dtype is correct
        with pytest.raises(RuntimeError):
            BVH.from_vertices(np.zeros((7, 4), dtype=np.float32))  # Not multiple of 3
        with pytest.raises(RuntimeError):
            BVH.from_indexed_mesh(np.zeros((4, 3), np.float32), np.zeros((2, 3), np.uint32))  # Verts not (V,4)

    def test_invalid_parameters(self):
        """Tests for invalid scalar parameters."""

        points = np.zeros((3, 3), dtype=np.float32)

        with pytest.raises(RuntimeError): BVH.from_points(points, radius=0.0)
        with pytest.raises(RuntimeError): BVH.from_points(points, radius=-1.0)


class TestCustomGeometry:
    def test_from_aabbs_intersection_and_uvs(self):
        """Tests BVH built from AABBs, including hit UVs"""

        aabbs = np.array([
            [[-1, -1, -0.1], [1, 1, 0.1]],  # AABB for first primitive
            [[2, 2, 4.9], [4, 4, 5.1]],  # AABB for second primitive
        ], dtype=np.float32)

        bvh = BVH.from_aabbs(aabbs)

        assert bvh.prim_count == 2

        # Hit center of -Z face
        ray = Ray(origin=(0, 0, -1), direction=(0, 0, 1))
        hit_dist = bvh.intersect(ray)

        # The first intersection with the box at z=[-0.1, 0.1] is at z=-0.1
        # Distance from z=-1 is 0.9
        assert np.isclose(hit_dist, 0.9)
        assert ray.prim_id == 0

        np.testing.assert_allclose((ray.u, ray.v), (0.5, 0.5))

    def test_from_points_intersection_and_uvs(self):
        """Tests BVH built from points (as spheres), including hit UVs"""

        points = np.array([[10, 10, 10]], dtype=np.float32)
        bvh = BVH.from_points(points, radius=0.5)

        # Hit front of sphere
        ray = Ray(origin=(10, 10, 0), direction=(0, 0, 1))
        bvh.intersect(ray)

        assert np.isclose(ray.t, 9.5)
        assert ray.prim_id == 0

        # Hit point is (10, 10, 9.5). Normal is (0,0,-1). u=0.25, v=0.5
        np.testing.assert_allclose((ray.u, ray.v), (0.25, 0.5))


# ==============================================================================
# VISUALISATION OF TEST SCENE
# ==============================================================================

def view_test_scene():
    """Builds and visualizes the TLAS test scene"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization demo.")
        return

    # Setup scene
    # BLAS 0: Unit cube
    cube_verts = np.zeros((8, 4), dtype=np.float32)
    cube_verts[:, :3] = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    cube_indices = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 7], [0, 7, 3],
        [1, 5, 6], [1, 6, 2], [0, 1, 5], [0, 5, 4], [3, 2, 6], [3, 6, 7]
    ], dtype=np.uint32)
    bvh_cube_blas = BVH.from_indexed_mesh(cube_verts, cube_indices)

    # BLAS 1: A 2x2 quad on the XY plane
    quad_verts = np.zeros((4, 4), dtype=np.float32)
    quad_verts[:, :3] = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    quad_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    bvh_quad_blas = BVH.from_indexed_mesh(quad_verts, quad_indices)

    blas_geometries = [
        (cube_verts[:, :3], cube_indices),
        (quad_verts[:, :3], quad_indices)
    ]
    blases = [bvh_cube_blas, bvh_quad_blas]

    # Define Instances
    instance_dtype = np.dtype([('transform', '<f4', (4, 4)), ('blas_id', '<u4'), ('mask', '<u4')])
    instances = np.zeros(4, dtype=instance_dtype)
    instances[0] = (np.identity(4, dtype=np.float32), 0, 0b0001)
    instances[1] = (translation_matrix(np.array([5, 0, 0])) @ scale_matrix(2.0), 0, 0b0010)
    instances[2] = (translation_matrix(np.array([0, 5, 0])) @ rotation_matrix(np.array([0, 1, 0]), np.pi / 2), 1,
                    0b0100)
    instances[3] = (translation_matrix(np.array([0, 0, -5])), 1, 0b1000)

    tlas = BVH.build_tlas(instances, blases)

    # Define rays for visualization
    ray_defs = [
        {'label': "Hit Inst 0 (Cube)", 'o': [0, 0, -2], 'd': [0, 0, 1], 'mask': 0xFFFFFFFF},
        {'label': "Hit Inst 1 (Scaled Cube)", 'o': [5, 0, -2], 'd': [0, 0, 1], 'mask': 0xFFFFFFFF},
        {'label': "Hit Inst 2 (Rotated Quad)", 'o': [-2, 5, 0], 'd': [1, 0, 0], 'mask': 0xFFFFFFFF},
        {'label': "Miss", 'o': [5, 5, 5], 'd': [1, 1, 1], 'mask': 0xFFFFFFFF},
        {'label': "Masked: Hit Inst 3 (Quad)", 'o': [0, 0, 2], 'd': [0, 0, -1], 'mask': 0b1000},
        {'label': "Unmasked: Hit Inst 0", 'o': [0, 0, 2], 'd': [0, 0, -1], 'mask': 0b0001},
    ]
    origins = np.array([r['o'] for r in ray_defs], dtype=np.float32)
    directions = np.array([r['d'] for r in ray_defs], dtype=np.float32)
    masks = np.array([r['mask'] for r in ray_defs], dtype=np.uint32)
    t_max = np.full(len(ray_defs), 100.0, dtype=np.float32)

    # Intersect rays
    hits = tlas.intersect_batch(origins, directions, t_max, masks)

    # Plotting
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    instance_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Muted blue, orange, green, red

    # Plot instance geometries
    for i, inst in enumerate(instances):
        verts, indices = blas_geometries[inst['blas_id']]
        transform = inst['transform']

        verts_h = np.ones((verts.shape[0], 4), dtype=np.float32)
        verts_h[:, :3] = verts
        transformed_verts_h = verts_h @ transform.T

        tris_to_plot = transformed_verts_h[:, :3][indices]
        ax.add_collection3d(Poly3DCollection(
            tris_to_plot,
            alpha=0.3,
            facecolor=instance_colors[i],
            edgecolor=instance_colors[i]
        ))

        # Label instances
        centroid = np.mean(transformed_verts_h[:, :3], axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], f"Inst {i}", color=instance_colors[i])

    # Plot TLAS root AABB
    plot_aabb(ax, tlas.aabb_min, tlas.aabb_max, color='purple', linestyle='-', label='TLAS Root AABB')

    # Plot instance AABBs (world space)
    for node in tlas.nodes:
        if node['prim_count'] > 0:  # It's a leaf node pointing to instances
            plot_aabb(ax, node['aabb_min'], node['aabb_max'], color='gray', linestyle=':')

    # Plot rays
    for i, (origin, direction, hit, ray_def) in enumerate(zip(origins, directions, hits, ray_defs)):
        is_hit = hit['inst_id'].astype(np.int32) != -1

        if is_hit:
            hit_point = origin + direction * hit['t']
            ax.plot(*zip(origin, hit_point), color='black', marker='o', markevery=[0],
                    label=ray_def['label'] if i == 0 else None)
            ax.scatter(*hit_point, color=instance_colors[hit['inst_id']], s=80, edgecolor='black', zorder=10)
            ax.text(*hit_point, f" Hit Inst {hit['inst_id']}", color='black', zorder=11)
        else:
            end_point = origin + direction * 20  # fixed length for misses
            ax.plot(*zip(origin, end_point), color='red', alpha=0.7, linestyle=':', marker='o', markevery=[0],
                    label=ray_def['label'] if i == 0 else None)
            ax.text(*end_point, " Miss", color='red', alpha=0.9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of TLAS Test Scene')

    center = (tlas.aabb_min + tlas.aabb_max) / 2
    size = np.max(tlas.aabb_max - tlas.aabb_min) * 1.2
    ax.set_xlim(center[0] - size / 2, center[0] + size / 2)
    ax.set_ylim(center[1] - size / 2, center[1] + size / 2)
    ax.set_zlim(center[2] - size / 2, center[2] + size / 2)

    plt.tight_layout()
    plt.show()


def plot_aabb(ax, aabb_min, aabb_max, **kwargs):
    """Plots a 3D bounding box"""
    points = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]], [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]], [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]], [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]], [aabb_min[0], aabb_max[1], aabb_max[2]],
    ])
    edges = [
        [points[0], points[1], points[2], points[3], points[0]],
        [points[4], points[5], points[6], points[7], points[4]],
        [points[0], points[4]], [points[1], points[5]],
        [points[2], points[6]], [points[3], points[7]],
    ]
    for edge in edges:
        line = np.array(edge)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], **kwargs)


if __name__ == "__main__":
    print("Running visualization demo...")
    view_test_scene()
    print("\nTo run the automated test suite, use 'pytest'")