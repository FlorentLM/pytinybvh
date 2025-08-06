import numpy as np
from pytinybvh import BVH, Ray, vec3

# Create some triangle geometry
triangles = np.array([
    [   # Tri 1
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0]
    ],

    [   # Tri 2
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0]
    ],
], dtype=np.float32)

# Option A: User-friendly (with copy)
bvh = BVH.from_triangles(triangles)

print("BVH built successfully!")
print("Node array shape:", bvh.nodes.shape)
print("Node array dtype:", bvh.nodes.dtype)

# Access fields by name
print("\nRoot node (index 0):")
root_node = bvh.nodes[0]
print("  AABB Min:", root_node['aabb_min'])
print("  AABB Max:", root_node['aabb_max'])
print("  Is Leaf:", root_node['prim_count'] > 0)
print("  Left/First:", root_node['left_first'])

# Access whole columns at once
all_tri_counts = bvh.nodes['prim_count']
leaf_nodes = bvh.nodes[all_tri_counts > 0]
print(f"\nFound {len(leaf_nodes)} leaf nodes.")

# -------

# Perform an intersection query
ray = Ray(
    origin=vec3(0.0, 0.0, -1.0),
    direction=vec3(0.0, 0.0, 1.0)
)
print("\nBefore intersect:", ray)

hit = bvh.intersect(ray)

print("Hit:", hit)
print("After intersect:", ray)
if hit:
    print(f"Intersection Details: t={ray.t:.3f}, prim_id={ray.prim_id}")