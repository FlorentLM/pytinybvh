import os
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from pytinybvh import BVH, Ray, BuildQuality

# ==============================================================================

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_aabb(ax, aabb_min, aabb_max, color='gray', linestyle='--', alpha=0.8):

    if not MATPLOTLIB_AVAILABLE:
        return

    points = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ])
    edges = [
        [points[0], points[1], points[2], points[3], points[0]],
        [points[4], points[5], points[6], points[7], points[4]],
        [points[0], points[4]],
        [points[1], points[5]],
        [points[2], points[6]],
        [points[3], points[7]],
    ]
    for edge in edges:
        line = np.array(edge)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color, linestyle=linestyle, alpha=alpha)


def plot_bvh_n_rays(bvh, source_triangles, origins, directions, hits, tmax_values):

    if not MATPLOTLIB_AVAILABLE:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Accent(np.linspace(0, 1, len(origins)))

    # Plot triangles
    poly_collection = Poly3DCollection(source_triangles, alpha=0.3, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(poly_collection)
    ax.text(0.0, 1.0, 0.1, "Tri 0", color='black')
    ax.text(3.0, 4.0, 5.1, "Tri 1", color='black')

    # Plot BVH nodes
    root_node = bvh.nodes[0]
    plot_aabb(ax, root_node['aabb_min'], root_node['aabb_max'], color='purple', linestyle='-', alpha=0.6)
    ax.text(root_node['aabb_min'][0], root_node['aabb_min'][1], root_node['aabb_max'][2], "Root BVH", color='purple')

    if not (root_node['prim_count'] > 0):  # If not a leaf
        left_child = bvh.nodes[root_node['left_first']]
        right_child = bvh.nodes[root_node['left_first'] + 1]
        plot_aabb(ax, left_child['aabb_min'], left_child['aabb_max'], color='orange', linestyle='--', alpha=0.5)
        plot_aabb(ax, right_child['aabb_min'], right_child['aabb_max'], color='orange', linestyle='--', alpha=0.5)

    # Plot rays
    for i in range(len(origins)):
        origin = origins[i]
        direction = directions[i]
        hit_record = hits[i]
        color = colors[i]
        tmax = tmax_values[i] * 2

        prim_id_as_int = hit_record['prim_id'].astype(np.int32)

        if prim_id_as_int != -1:  # Hit
            hit_point = origin + direction * hit_record['t']
            ax.plot([origin[0], hit_point[0]], [origin[1], hit_point[1]], [origin[2], hit_point[2]],
                    color=color, marker='o', markevery=[0], markersize=5,
                    label=f'Ray {i} (Hit @ t={hit_record["t"]:.1f})')
            ax.scatter(hit_point[0], hit_point[1], hit_point[2], color=color, s=60, edgecolor='black')
        else:  # Miss
            # slightly desaturated color for miss
            end_point = origin + direction * min(tmax, 20)

            ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]],
                    color=color, alpha=0.9, linestyle=':', marker='o', markevery=[0], markersize=5,
                    label=f'Ray {i} (Miss)')
            ax.scatter(end_point[0], end_point[1], end_point[2], color=color, alpha=0.9, marker='x', s=50)

    # ︵‿︵‿୨♡୧‿︵‿︵ aesthetics ︵‿︵‿୨♡୧‿︵‿︵
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('BVH Visualization with Ray Intersections')
    ax.auto_scale_xyz(  # auto-scale axes to fit main BVH
        [bvh.aabb_min[0], bvh.aabb_max[0]],
        [bvh.aabb_min[1], bvh.aabb_max[1]],
        [bvh.aabb_min[2], bvh.aabb_max[2]]
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=(0, 0, 0.85, 1))  # adjust layout to make space for legend
    plt.show()


# ==============================================================================


# ==============================================================================
# --- BUILD QUALITY TESTS ---
# ==============================================================================


# Create some triangle geometry
triangles = np.array([
    # Triangle 0 (around origin, at z=0)
    [[-1.0, -1.0, 0.0],
     [ 1.0, -1.0, 0.0],
     [ 0.0,  1.0, 0.0]],
    # Triangle 1 (around (3, 3), at z=5)
    [[ 2.0,  2.0, 5.0],
     [ 4.0,  2.0, 5.0],
     [ 3.0,  4.0, 5.0]],
], dtype=np.float32)

print("\n--- Testing Build Quality (from_triangles) ---")
bvh_balanced = BVH.from_triangles(triangles, quality=BuildQuality.Balanced)
bvh_high = BVH.from_triangles(triangles, quality=BuildQuality.High)
bvh_quick = BVH.from_triangles(triangles, quality=BuildQuality.Quick)

assert(bvh_balanced.nodes.nbytes == 128)
assert(bvh_high.nodes.nbytes == 128)
assert(bvh_quick.nodes.nbytes == 128)

assert bvh_balanced.quality == BuildQuality.Balanced
assert bvh_high.quality == BuildQuality.High
assert bvh_quick.quality == BuildQuality.Quick

print("BVH correctly built with Balanced quality")
print("BVH correctly built with High quality")
print("BVH correctly built with Quick quality")

try:
    bvh_high.refit()
    # This should not be reached
    assert False, "Refit should have raised an exception for a High quality BVH."
except RuntimeError as e:
    print(f"Successfully caught expected error on refit: {e}")

print("\nBuild quality tests successful!")


# ==============================================================================
# --- CORE ZERO-COPY BUILDER TESTS ---
# ==============================================================================

print("\n--- Testing Core (Zero-Copy) Builders ---")

# --- from_vertices Test ---
print("\nTesting from_vertices...")
vertices_4d = np.zeros((6, 4), dtype=np.float32)
vertices_4d[:, :3] = triangles.reshape(6, 3)
bvh_vertices = BVH.from_vertices(vertices_4d)
assert bvh_vertices.prim_count == 2
ray_test_vertices = Ray(origin=(0,0,-1), direction=(0,0,1))
bvh_vertices.intersect(ray_test_vertices)
assert np.isclose(ray_test_vertices.t, 1.0)
print("BVH from_vertices successful.")


# --- from_indexed_mesh Test ---
print("\nTesting from_indexed_mesh...")
indexed_vertices_3d = np.array([[0,0,10], [5,0,10], [0,5,10], [5,5,10]], dtype=np.float32)
indexed_vertices = np.zeros((4, 4), dtype=np.float32)
indexed_vertices[:, :3] = indexed_vertices_3d
indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
bvh_indexed = BVH.from_indexed_mesh(indexed_vertices, indices)
assert bvh_indexed.prim_count == 2
ray_test_indexed = Ray(origin=(2.5, 2.5, 0), direction=(0, 0, 1))
bvh_indexed.intersect(ray_test_indexed)
assert np.isclose(ray_test_indexed.t, 10.0)
print("BVH from_indexed_mesh successful.")


# Test refitting with indexed mesh
print("Testing refit on indexed mesh...")
indexed_vertices[:, 2] = 20.0 # Move all vertices to z=20
bvh_indexed.refit()
ray_refit = Ray(origin=(2.5, 2.5, 0.0), direction=(0.0, 0.0, 1.0))
hit_dist_refit = bvh_indexed.intersect(ray_refit)
print(f"Ray aimed at *refit* indexed quad: Returned t={hit_dist_refit:.3f}, Ray object: {ray_refit}")
assert np.isclose(hit_dist_refit, 20.0)
print("Indexed mesh refit successful.")


# --- from_aabbs Test ---
print("\nTesting from_aabbs...")
aabbs = np.array([
    [[-1, -1, -0.1], [1, 1, 0.1]], # AABB for first primitive
    [[2, 2, 4.9], [4, 4, 5.1]],    # AABB for second primitive
], dtype=np.float32)

bvh_aabbs = BVH.from_aabbs(aabbs)
assert bvh_aabbs.prim_count == 2

# Test intersection with the first AABB
ray_test_aabbs = Ray(origin=(0,0,-1), direction=(0,0,1))
hit_dist = bvh_aabbs.intersect(ray_test_aabbs)
print(f"Ray aimed at AABB 0: Returned t={hit_dist:.3f}, Ray object: {ray_test_aabbs}")

assert np.isclose(hit_dist, 0.9)
assert ray_test_aabbs.prim_id == 0

# Test AABB UV coordinates
print(f"  Hit UVs: ({ray_test_aabbs.u:.3f}, {ray_test_aabbs.v:.3f}) -> Expected: (0.500, 0.500)")
assert np.isclose(ray_test_aabbs.u, 0.5)
assert np.isclose(ray_test_aabbs.v, 0.5)

print("BVH from_aabbs successful.")


print("Testing occlusion for AABB BVH...")
# Ray that is occluded by the first AABB (hit at t=0.9, max t=100)
ray_occluded_aabb = Ray(origin=(0,0,-1), direction=(0,0,1), t=100.0)
assert bvh_aabbs.is_occluded(ray_occluded_aabb)
print("  - Occluded ray test: PASSED")

# Ray that is aimed at the AABB but has t_max too short (hit at t=0.9, max t=0.5)
ray_not_occluded_aabb = Ray(origin=(0,0,-1), direction=(0,0,1), t=0.5)
assert not bvh_aabbs.is_occluded(ray_not_occluded_aabb)
print("  - t_max limited ray test: PASSED")

# Batch occlusion test for AABBs
aabb_origins = np.array([
    [0, 0, -1],     # Should be occluded
    [0, 0, -1],     # Should NOT be occluded (t_max too short)
    [10, 10, -1],   # Should miss
], dtype=np.float32)
aabb_directions = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=np.float32)
aabb_t_max = np.array([100.0, 0.5, 100.0], dtype=np.float32)

occluded_results_aabb = bvh_aabbs.is_occluded_batch(aabb_origins, aabb_directions, aabb_t_max)
expected_occlusion_aabb = np.array([True, False, False])
np.testing.assert_array_equal(occluded_results_aabb, expected_occlusion_aabb)
print("  - Batch occlusion test: PASSED")

# ==============================================================================
# --- CONVENIENCE BUILDER TESTS ---
# ==============================================================================
print("\n--- Testing Convenience (Copying) Builders ---")

# --- from_points Test ---
print("\nTesting from_points...")
points = np.array([[10,10,10], [20,20,20]], dtype=np.float32)
bvh_points = BVH.from_points(points, radius=0.5)
assert bvh_points.prim_count == 2

# Hit the first point's sphere
ray_test_points = Ray(origin=(10,10,0), direction=(0.0, 0.0, 1.0))
hit_dist = bvh_points.intersect(ray_test_points)
print(f"Ray aimed at sphere 0: Returned t={hit_dist:.3f}, Ray object: {ray_test_points}")

assert np.isclose(ray_test_points.t, 9.5) # Hits at z = 10 - 0.5
assert ray_test_points.prim_id == 0

# Test Sphere UV coordinates
print(f"  Hit UVs: ({ray_test_points.u:.3f}, {ray_test_points.v:.3f}) -> Expected: (0.250, 0.500)")
assert np.isclose(ray_test_points.u, 0.25)
assert np.isclose(ray_test_points.v, 0.5)

print("BVH from_points successful.")

print("Testing occlusion for sphere BVH...")
# This ray is aimed through the center of the sphere at (10,10,10) with radius 0.5.
# It should be occluded. The hit is at t=9.5, ray.t=100.
ray_occluded_sphere = Ray(origin=(10,10,0), direction=(0,0,1), t=100.0)
assert bvh_points.is_occluded(ray_occluded_sphere)
print("  - Occluded ray test: PASSED")

# This ray is aimed at the sphere but its max distance is too short.
# The hit is at t=9.5, but ray.t=5.0, so it should not be occluded.
ray_not_occluded_sphere = Ray(origin=(10,10,0), direction=(0,0,1), t=5.0)
assert not bvh_points.is_occluded(ray_not_occluded_sphere)
print("  - t_max limited ray test: PASSED")

# A ray that grazes the sphere should be a hit
# Sphere center (10,10,10), radius 0.5. Ray origin (10.4, 10, 0)
ray_grazing_sphere = Ray(origin=(10.4, 10, 0), direction=(0,0,1), t=100.0)
assert bvh_points.is_occluded(ray_grazing_sphere)
print("  - Grazing ray test: PASSED")

# A ray that just misses the sphere
ray_missing_sphere = Ray(origin=(10.6, 10, 0), direction=(0,0,1), t=100.0)
assert not bvh_points.is_occluded(ray_missing_sphere)
print("  - Missing ray test: PASSED")

# ==============================================================================
# --- SAVE/LOAD TEST ---
# ==============================================================================

print("\n--- Testing Saving / Loading (Indexed and Non-Indexed) ---")
bvh_soup = BVH.from_vertices(vertices_4d) # zero-copy builder
bvh_soup.save("bvh_soup_test.bvh")
assert os.path.exists("bvh_soup_test.bvh")
print(f'Saved soup file size: {os.stat("bvh_soup_test.bvh").st_size} bytes')
bvh_loaded_soup = BVH.load("bvh_soup_test.bvh", vertices_4d)
os.remove("bvh_soup_test.bvh")
assert np.all(bvh_loaded_soup.nodes == bvh_soup.nodes)
assert np.all(bvh_loaded_soup.prim_indices == bvh_soup.prim_indices)
print("Non-indexed BVH saved and loaded correctly.")

bvh_indexed.save(Path("bvh_indexed_test.bvh"))
assert os.path.exists("bvh_indexed_test.bvh")
print(f'Saved indexed file size: {os.stat("bvh_indexed_test.bvh").st_size} bytes')
bvh_loaded_indexed = BVH.load("bvh_indexed_test.bvh", indexed_vertices, indices)
os.remove("bvh_indexed_test.bvh")
assert np.all(bvh_loaded_indexed.nodes == bvh_indexed.nodes)
assert np.all(bvh_loaded_indexed.prim_indices == bvh_indexed.prim_indices)
print("Indexed BVH saved and loaded correctly.")

print("\nSave/load tests successful!")



# ==============================================================================

# Use the loaded soup BVH for subsequent tests for consistency
bvh = bvh_loaded_soup


# ==============================================================================
# --- BVH STRUCTURE TESTS ---
# ==============================================================================

print("\n--- Testing Structure ---")

# Access fields by name
root_node = bvh.nodes[0]

print("\nRoot node (index 0):")
print("  AABB Min :", root_node['aabb_min'])
print("  AABB Max :", root_node['aabb_max'])
print("  Is Leaf  :", root_node['prim_count'] > 0)
print(f"  Children : {root_node['left_first']} and {root_node['left_first'] + 1}")

# Access whole columns at once
all_tri_counts = bvh.nodes['prim_count']
leaf_nodes = bvh.nodes[all_tri_counts > 0]

assert len(bvh.nodes) == 4
assert len(leaf_nodes) == 2

print(f"\nFound {len(bvh.nodes)} nodes, including {len(leaf_nodes)} leaf nodes.")
print('\nStructure tests successful!')


# ==============================================================================
# --- RAY INTERSECTION TESTS ---
# ==============================================================================

print("\n--- Testing single ray intersect ---")
# This ray should hit the first triangle (ID 0) at t=10
ray_hit = Ray(
    origin=(0.0, 0.0, -10.0),
    direction=(0.0, 0.0, 1.0)
)
hit_distance = bvh.intersect(ray_hit)
print(f"Ray 1 (expect hit): Returned t={hit_distance:.3f}, Ray object updated to: {ray_hit}")
assert not np.isinf(hit_distance)
assert np.isclose(hit_distance, 10.0)
assert ray_hit.prim_id == 0 and np.isclose(ray_hit.t, 10.0)

# This ray should miss everything
ray_miss = Ray(
    origin=(10.0, 10.0, -10.0),
    direction=(0.0, 0.0, 1.0)
)
hit_distance_miss = bvh.intersect(ray_miss)
print(f"Ray 2 (expect miss): Returned t={hit_distance_miss:.3f}, Ray object state: {ray_miss}")
assert np.isinf(hit_distance_miss)
assert ray_miss.prim_id == np.iinfo(np.uint32).max # Check that the ray was not modified


print("\n--- Barycentric check ---")
ray_bary_check = Ray(origin=(0, 0, -1), direction=(0, 0, 1))
bvh.intersect(ray_bary_check)

print(f"Ray barycentric check: u={ray_bary_check.u:.3f}, v={ray_bary_check.v:.3f} -> Expected: (0.333, 0.333)")
assert np.isclose(ray_bary_check.u, 0.25)
assert np.isclose(ray_bary_check.v, 0.5)
print("Barycentric coordinate test successful!")


print("\n--- Testing batch ray intersect ---")
# Prepare batch of rays with staggered origins for better visualization
origins = np.array([
    [0.1, 0.1, -10.0],   # Ray 0: Aimed at tri 0. Should hit at t=10
    [3.1, 2.9, -10.0],   # Ray 1: Aimed at tri 1. Should hit at t=15
    [4.5, 4.5, -10.0],    # Ray 2: Aimed at nothing. Should miss.
    [0.2, -0.2, -10.0],  # Ray 3: Aimed at tri 0, but will be stopped by t_max
    [1.5, 1.5, -10.0],  # Ray 4: Aimed between the two triangles. Hits the bounding boxes but not the prims.
], dtype=np.float32)

directions = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# Optional t_max array. Ray 3 has t_max=4, so it won't reach tri 0 at t=10
t_max = np.array([100.0, 100.0, 100.0, 4.0, 100.0], dtype=np.float32)

# Perform the batch intersection
hits = bvh.intersect_batch(origins, directions, t_max)

print("Batch intersection results (structured array):")
print(hits)
print("\nIndividual hit results:")
print(f"Hit 0 (prim_id, t): ({hits[0]['prim_id'].astype(np.int32)}, {hits[0]['t']:.3f}) -> Expected: (0, 10.0)")
print(f"Hit 1 (prim_id, t): ({hits[1]['prim_id'].astype(np.int32)}, {hits[1]['t']:.3f}) -> Expected: (1, 15.0)")
print(f"Hit 2 (prim_id, t): ({hits[2]['prim_id'].astype(np.int32)}, {hits[2]['t']:.3f}) -> Expected: (-1, inf)")
print(f"Hit 3 (prim_id, t): ({hits[3]['prim_id'].astype(np.int32)}, {hits[3]['t']:.3f}) -> Expected: (-1, inf) due to t_max")
print(f"Hit 4 (prim_id, t): ({hits[4]['prim_id'].astype(np.int32)}, {hits[4]['t']:.3f}) -> Expected: (-1, inf)")

# Assert for hit on triangle 0
assert hits[0]['prim_id'] == 0 and np.isclose(hits[0]['t'], 10.0)
# Assert for hit on triangle 1
assert hits[1]['prim_id'] == 1 and np.isclose(hits[1]['t'], 15.0)
prim_ids_as_signed = hits['prim_id'].astype(np.int32)
assert prim_ids_as_signed[2] == -1 and np.isinf(hits[2]['t'])
assert prim_ids_as_signed[3] == -1 and np.isinf(hits[3]['t'])
assert prim_ids_as_signed[4] == -1 and np.isinf(hits[4]['t'])

print("\nBatch test successful!")


# ==============================================================================
# --- RAY OCCLUSION TESTS ---
# ==============================================================================


print("\n--- Testing single ray occlusion ---")
# This ray is occluded by the second triangle (at z=5)
# Note: ray.t=100 defines the max distance for the occlusion check. The hit is at t=15.
ray_occluded = Ray(
    origin=(3.0, 3.0, -10.0),
    direction=(0.0, 0.0, 1.0),
    t=100.0
)
is_occluded = bvh.is_occluded(ray_occluded)
print(f"Ray 1 (expect occluded): {is_occluded}")
assert is_occluded

# This ray misses everything
ray_not_occluded = Ray(
    origin=(10.0, 10.0, -10.0),
    direction=(0.0, 0.0, 1.0)
)
is_occluded_miss = bvh.is_occluded(ray_not_occluded)
print(f"Ray 2 (expect not occluded): {is_occluded_miss}")
assert not is_occluded_miss

# This ray is aimed at the second triangle, but its max distance (t) is too short.
# The hit is at t=15, but we only check up to t=4.0, so it should NOT be considered occluded.
ray_t_limited = Ray(
    origin=(3.0, 3.0, -10.0),
    direction=(0.0, 0.0, 1.0),
    t=4.0
)
is_occluded_t_limited = bvh.is_occluded(ray_t_limited)
print(f"Ray 3 (expect not occluded due to t_max): {is_occluded_t_limited}")
assert not is_occluded_t_limited

print("\n--- Testing batch ray occlusion ---")

# We use the same origins and directions as the intersect_batch test
# The t_max array now determines the occlusion distance
occlusion_t_max = np.array([
    100.0,  # Ray 0: Will be occluded by tri 0 (t=10)
    4.0,    # Ray 1: Will NOT be occluded by tri 1 (t=15) because t_max is too short
    100.0,  # Ray 2: Will not be occluded (misses everything)
], dtype=np.float32)

occluded_results = bvh.is_occluded_batch(origins[:3], directions[:3], occlusion_t_max)
expected_occlusion = np.array([True, False, False])

print("Batch occlusion results (boolean array):", occluded_results)
print("Expected results:                     ", expected_occlusion)

np.testing.assert_array_equal(occluded_results, expected_occlusion)

print("\nOcclusion tests successful!")

# ================================== DONE ======================================

print("\n\nAll tests passed!")

if MATPLOTLIB_AVAILABLE:
    plot_bvh_n_rays(
        bvh=bvh,
        source_triangles=triangles,
        origins=origins,
        directions=directions,
        hits=hits,
        tmax_values=t_max,
    )