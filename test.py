import time
import sys
from pathlib import Path
import numpy as np
import pytinybvh


def load_primitives_from_file(filepath: Path, points_only: bool) -> np.ndarray:

    import trimesh
    try:
        print('Loading test scene with trimesh...')
        data = trimesh.load(filepath)

        if points_only:
            return data.vertices.astype(np.float32)
        else:
            if not isinstance(data, trimesh.Trimesh):
                print(f"Warning: '{filepath}' is not a triangular mesh. Attempting to triangulate.")
                data = data.triangulate()
            if not hasattr(data, 'faces') or len(data.faces) == 0:
                print(f"Error: Could not load triangles from '{filepath}'. No face information found.")
                return np.array([], dtype=np.float32)
            triangles = data.vertices[data.faces]
            return triangles.reshape(-1, 9).astype(np.float32)

    except Exception as e:
        print(f"An error occurred while loading the file with trimesh: {e}")
        return np.array([], dtype=np.float32)


def generate_fallback_scene(points_only: bool) -> np.ndarray:

    if points_only:
        print("Generating a fallback scene: a sphere of points.")
        points = (np.random.randn(5000, 3))
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
        return points.astype(np.float32)
    else:
        print("Generating a fallback scene: a ground plane and a cube.")
        ground_plane = [[-10.0, -1.0, -10.0, 10.0, -1.0, -10.0, -10.0, -1.0, 10.0],
                        [10.0, -1.0, -10.0, 10.0, -1.0, 10.0, -10.0, -1.0, 10.0]]
        v = [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 1.5, -0.5], [-0.5, 1.5, -0.5], [-0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [-0.5, 1.5, 0.5]]
        cube_tris = [v[0] + v[1] + v[2], v[0] + v[2] + v[3], v[4] + v[5] + v[6], v[4] + v[6] + v[7], v[0] + v[4] + v[7],
                     v[0] + v[7] + v[3], v[1] + v[5] + v[6], v[1] + v[6] + v[2], v[3] + v[2] + v[6], v[3] + v[6] + v[7],
                     v[0] + v[1] + v[5], v[0] + v[5] + v[4]]
        return np.array(ground_plane + cube_tris, dtype=np.float32)


def view_bvh(primitives_np: np.ndarray, bvh_nodes: np.ndarray, max_depth: int = 6):

    import pyvista as pv
    prim_type = "points" if primitives_np.shape[1] == 3 else "triangles"

    plotter = pv.Plotter(window_size=[1600, 900])
    plotter.set_background('white')

    # Add original geometry
    if prim_type == "triangles":
        vertices = primitives_np.reshape(-1, 3)
        faces = np.hstack((np.full((len(vertices) // 3, 1), 3), np.arange(len(vertices)).reshape(-1, 3)))
        mesh = pv.PolyData(vertices, faces)
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black', line_width=1)
    else:  # points
        plotter.add_points(primitives_np, color='blue', point_size=5, render_points_as_spheres=True)

    # Recursively add the BVH boxes
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'purple', 'magenta']

    def add_node_to_plotter(node_index, depth):
        if depth >= max_depth: return

        node_data_bytes = bvh_nodes[node_index].tobytes()
        node_uint32 = np.frombuffer(node_data_bytes, dtype=np.uint32)
        node_float32 = np.frombuffer(node_data_bytes, dtype=np.float32)

        min_b, max_b = node_float32[0:3], node_float32[4:7]
        left_first_idx, prim_count = node_uint32[3], node_uint32[7]

        box = pv.Box(bounds=[min_b[0], max_b[0], min_b[1], max_b[1], min_b[2], max_b[2]])
        plotter.add_mesh(box, style='wireframe', color=colors[depth % len(colors)], line_width=2)

        if prim_count == 0:
            add_node_to_plotter(left_first_idx, depth + 1)
            add_node_to_plotter(left_first_idx + 1, depth + 1)

    print(f"\nVisualizing BVH up to depth {max_depth}...")
    add_node_to_plotter(node_index=0, depth=0)
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.show()


if __name__ == "__main__":

    TEST_SCENE_FILE = Path('dragon.ply')
    POINTS_ONLY = False     # True for point clouds, False for triangle meshes
    VISUALISE = True        # True to show the PyVista debug viewer
    VISUALISE_DEPTH = 7     # Viewer max depth

    if TEST_SCENE_FILE.exists():
        try:
            import trimesh
        except ImportError:
            print(f"\nError: Configuration requires loading '{TEST_SCENE_FILE}', but `trimesh` is not installed.")
            print("Please install it with: `uv pip install trimesh`")
            sys.exit(1)

    if VISUALISE:
        try:
            import pyvista
        except ImportError:
            print("\nError: Configuration has `VISUALISE = True`, but `pyvista` is not installed.")
            print("Please install it with: `uv pip install pyvista`")
            sys.exit(1)

    prims_np = np.array([], dtype=np.float32)

    if TEST_SCENE_FILE.exists():
        prims_np = load_primitives_from_file(TEST_SCENE_FILE, points_only=POINTS_ONLY)
        if prims_np.size > 0:
            print(f"Loaded {len(prims_np)} primitives from {TEST_SCENE_FILE}.")
    else:
        print(f"File '{TEST_SCENE_FILE}' not found.")
        prims_np = generate_fallback_scene(points_only=POINTS_ONLY)

    if prims_np.size == 0:
        print("No primitives to process. Exiting.")
        sys.exit(1)

    print(f"\nUsing numpy array `prims_np` of shape {prims_np.shape}")

    if prims_np.shape[1] == 9:
        prim_type = "triangles"
        build_func = pytinybvh.from_triangles
    elif prims_np.shape[1] == 3:
        prim_type = "points"
        build_func = pytinybvh.from_points
    else:
        print(f"Error: Unsupported primitive shape {prims_np.shape}. Must be (N, 3) or (N, 9).")
        sys.exit(1)

    # BVH building
    print("\nBuilding BVH using pytinybvh...")
    start_time = time.time_ns()
    bvh_nodes, prim_indices = build_func(prims_np)
    gen_time = time.time_ns() - start_time
    print(f"BVH build complete!")

    # Reordering primitives for rendering (simulates what a shader would need)
    start_time = time.time_ns()
    reordered_primitives = prims_np[prim_indices]
    reorder_time = time.time_ns() - start_time

    print(f"\nBVH built for {len(prims_np)} {prim_type}.")
    print(f"  - Build      : {gen_time / 1e6:.4f} ms")
    print(f"  - Reordering : {reorder_time / 1e6:.4f} ms")
    print(f"  - Total      : {(gen_time + reorder_time) / 1e6:.2f} ms")
    print(f"    (could be recomputed {1 / ((gen_time + reorder_time) / 1e9):.2f} times per second)")

    if VISUALISE:
        view_bvh(prims_np, bvh_nodes, max_depth=VISUALISE_DEPTH)