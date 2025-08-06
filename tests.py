import numpy as np
np.set_printoptions(precision=3, suppress=True)
from pytinybvh import BVH, Ray, vec3

# Create some triangle geometry
triangles = np.array([
    # Triangle 0 (around origin, at z=0)
    [[-1.0, -1.0, 0.0],
     [ 1.0, -1.0, 0.0],
     [ 0.0,  1.0, 0.0]],
    # Triangle 1 (around (3,3), at z=5)
    [[ 2.0,  2.0, 5.0],
     [ 4.0,  2.0, 5.0],
     [ 3.0,  4.0, 5.0]],
], dtype=np.float32)

bvh = BVH.from_triangles(triangles)
print("BVH built successfully!")

print("Nodes array shape:", bvh.nodes.shape)
print("Nodes array dtype:", bvh.nodes.dtype)

# Access fields by name
root_node = bvh.nodes[1]

print("\nRoot node (index 0):")
print("  AABB Min :", root_node['aabb_min'])
print("  AABB Max :", root_node['aabb_max'])
print("  Is Leaf  :", root_node['prim_count'] > 0)
print(f"  Children : {root_node['left_first']} and {root_node['left_first'] + 1}")

# Access whole columns at once
all_tri_counts = bvh.nodes['prim_count']
leaf_nodes = bvh.nodes[all_tri_counts > 0]
print(f"\nFound {len(leaf_nodes)} leaf nodes.")

# -------

print("\n--- Testing single ray intersect ---")
# This ray should hit the first triangle (ID 0) at t=10
ray_hit = Ray(
    origin=vec3(0.0, 0.0, -10.0),
    direction=vec3(0.0, 0.0, 1.0)
)
hit_distance = bvh.intersect(ray_hit)
print(f"Ray 1 (expect hit): Returned t={hit_distance:.3f}, Ray object updated to: {ray_hit}")
assert not np.isinf(hit_distance)
assert np.isclose(hit_distance, 10.0)
assert ray_hit.prim_id == 0 and np.isclose(ray_hit.t, 10.0)

# This ray should miss everything
ray_miss = Ray(
    origin=vec3(10.0, 10.0, -10.0),
    direction=vec3(0.0, 0.0, 1.0)
)
hit_distance_miss = bvh.intersect(ray_miss)
print(f"Ray 2 (expect miss): Returned t={hit_distance_miss:.3f}, Ray object state: {ray_miss}")
assert np.isinf(hit_distance_miss)
assert ray_miss.prim_id == np.iinfo(np.uint32).max # Check that the ray was not modified

print("\n--- Testing batch ray intersect ---")
# Prepare batch of rays specifically aimed at their targets
origins = np.array([
    [0.0, 0.0, -10.0],   # Ray 0: Aimed at tri 0. Should hit at t=10.
    [3.0, 3.0, -10.0],   # Ray 1: Aimed at tri 1. Should hit at t=15.
    [10.0, 10.0, -10.0], # Ray 2: Aimed at nothing. Should miss.
    [0.0, 0.0, -10.0],   # Ray 3: Aimed at tri 0, but will be stopped by t_max.
], dtype=np.float32)

directions = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# Optional t_max array. Ray 3 has t_max=4, so it won't reach tri 0 at t=10
t_max = np.array([100.0, 100.0, 100.0, 4.0], dtype=np.float32)

# Perform the batch intersection
hits = bvh.intersect_batch(origins, directions, t_max)

print("Batch intersection results (structured array):")
print(hits)
print("\nIndividual hit results:")
print(f"Hit 0 (prim_id, t): ({hits[0]['prim_id'].astype(np.int32)}, {hits[0]['t']:.3f}) -> Expected: (0, 10.0)")
print(f"Hit 1 (prim_id, t): ({hits[1]['prim_id'].astype(np.int32)}, {hits[1]['t']:.3f}) -> Expected: (1, 15.0)")
print(f"Hit 2 (prim_id, t): ({hits[2]['prim_id'].astype(np.int32)}, {hits[2]['t']:.3f}) -> Expected: (-1, inf)")
print(f"Hit 3 (prim_id, t): ({hits[3]['prim_id'].astype(np.int32)}, {hits[3]['t']:.3f}) -> Expected: (-1, inf) due to t_max")

# Assert for hit on triangle 0
assert hits[0]['prim_id'] == 0 and np.isclose(hits[0]['t'], 10.0)
# Assert for hit on triangle 1
assert hits[1]['prim_id'] == 1 and np.isclose(hits[1]['t'], 15.0)
prim_ids_as_signed = hits['prim_id'].astype(np.int32)
assert prim_ids_as_signed[2] == -1 and np.isinf(hits[2]['t'])
assert prim_ids_as_signed[3] == -1 and np.isinf(hits[3]['t'])

print("\nBatch test successful!")

# -------


print("\n--- Testing single ray occlusion ---")
# This ray is occluded by the second triangle (at z=5)
# Note: ray.t=100 defines the max distance for the occlusion check. The hit is at t=15.
ray_occluded = Ray(
    origin=vec3(3.0, 3.0, -10.0),
    direction=vec3(0.0, 0.0, 1.0),
    t=100.0
)
is_occluded = bvh.is_occluded(ray_occluded)
print(f"Ray 1 (expect occluded): {is_occluded}")
assert is_occluded

# This ray misses everything
ray_not_occluded = Ray(
    origin=vec3(10.0, 10.0, -10.0),
    direction=vec3(0.0, 0.0, 1.0)
)
is_occluded_miss = bvh.is_occluded(ray_not_occluded)
print(f"Ray 2 (expect not occluded): {is_occluded_miss}")
assert not is_occluded_miss

# This ray is aimed at the second triangle, but its max distance (t) is too short.
# The hit is at t=15, but we only check up to t=4.0, so it should NOT be considered occluded.
ray_t_limited = Ray(
    origin=vec3(3.0, 3.0, -10.0),
    direction=vec3(0.0, 0.0, 1.0),
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