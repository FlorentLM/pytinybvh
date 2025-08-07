# pytinybvh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple Python bindings for the great C++ Bounding Volume Hierarchy (BVH) library [tinybvh](https://github.com/jbikker/tinybvh) by Jacco Bikker.

Exposes `tinybvh`'s fast BVH construction algorithms to Python, to use for CPU-side BVH generation in Python applications.
Can be used for real-time ray tracing with PyOpenGL or Vulkan, or CPU-side stuff like collision detection, etc.

**Note:** Not all functionalities are implemented yet!

## Prerequisites

Before installing, you need a few tools on your system:

1.  **A C++20 Compatible Compiler:**
    -   **Windows:** MSVC (Visual Studio 2019 or newer with the "Desktop development with C++" workload)
    -   **Linux:** GCC 10+ or Clang 11+
    -   **macOS:** Xcode with modern Clang
2.  **Python 3.4+**
3.  **Git:** Required for cloning the repository and its dependencies

## Installation

The C++ dependency (`tinybvh`) is included as a Git submodule.

1.  **Clone the repository:**
    ```bash
    git clone https://FlorentLM/pytinybvh.git
    cd pytinybvh
    ```

2.  **Initialize the C++ Submodule:**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Create and activate a virtual environment:**
    I like `uv`, but `venv` works too.
    ```bash
    # Using uv
    uv venv
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

4.  **Compile and install:**
    Run `uv sync`. This will automatically download Python dependencies and compile the C++ extension module.
    ```bash
    uv sync
    ```

If the process completes without errors, the `pytinybvh` module is now installed and ready to be used in your virtual environment.
5. **Use in other projects:**
    From the virtual environment of your other project, run the installation of `pytinybvh` in editable mode.
    ```bash
   # Still with uv
    uv pip install -e /path/to/this/repo
    ```
    
## Usage examples

`pytinybvh` provides two types of BVH builders: **Core (zero-copy)** builders for performance and **Convenience (include copying)** builders for ease of use with common data formats.

---

### Core builders (Zero-copy)

These methods are the most performant as they directly use the memory buffer of the passed numpy arrays without creating copies.
**You must ensure the source numpy arrays are not garbage-collected while the BVH is in use.**

#### `BVH.from_vertices(vertices, quality)`
Builds a BVH from a flat "triangle soup" array in `tinybvh`'s native format.

```python
import numpy as np
from pytinybvh import BVH, BuildQuality

# Each triangle is 3 consecutive vertices. Each vertex is (x, y, z, w).
# The 4th component (w) is only for alignment and is ignored.
vertices_4d = np.array([
    # Triangle 0
    [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
    # Triangle 1
    [2, 2, 2, 0], [3, 2, 2, 0], [2, 3, 2, 0],
], dtype=np.float32)

# This is a zero-copy operation.
bvh = BVH.from_vertices(vertices_4d, quality=BuildQuality.Balanced)
```

#### `BVH.from_indexed_mesh(vertices, indices, quality)`
The most memory-efficient method for triangle meshes. It also supports fast refitting if the vertex positions change.

```python
import numpy as np
from pytinybvh import BVH, BuildQuality

# 4 unique vertices
vertices_4d = np.array([
    [0, 0, 0, 0], # 0
    [1, 0, 0, 0], # 1
    [1, 1, 0, 0], # 2
    [0, 1, 0, 0], # 3
], dtype=np.float32)

# 2 triangles referencing the vertices
indices_u32 = np.array([
    [0, 1, 3],
    [1, 2, 3],
], dtype=np.uint32)

# This is a zero-copy operation.
bvh = BVH.from_indexed_mesh(vertices_4d, indices_u32)

# If you modify the vertices_4d array in-place...
vertices_4d[:, 2] += 5.0 # Move the mesh up by 5 units

# ...you can refit the BVH much faster than rebuilding.
bvh.refit()
```

#### `BVH.from_aabbs(aabbs, quality)`
Builds a BVH from a list of pre-defined Axis-Aligned Bounding Boxes. Useful for custom geometry, or for building a TLAS.

```python
import numpy as np
from pytinybvh import BVH, BuildQuality

# 2 AABBs defined by their min and max corners
aabbs_np = np.array([
    # AABB 0: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    [[-1, -1, -1], [1, 1, 1]],
    # AABB 1
    [[5, 5, 5], [6, 6, 6]],
], dtype=np.float32)

# This is a zero-copy operation.
# Note: Intersections will be against degenerate triangles representing the AABBs.
bvh = BVH.from_aabbs(aabbs_np)
```

---

### Convenience builders (with copying)

These methods accept more common numpy array layouts. They internally perform a one-time copy and reformatting of the data.

#### `BVH.from_triangles(triangles, quality)`
The easiest way to build a BVH from a list of triangles.

```python
import numpy as np
from pytinybvh import BVH

# An array of shape (N, 3, 3), or an array of shape (N, 9)
triangles_np = np.array([
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    [[2, 2, 2], [3, 2, 2], [2, 3, 2]],
], dtype=np.float32)

# This copies the data into the required format.
bvh = BVH.from_triangles(triangles_np)
```

#### `BVH.from_points(points, radius, quality)`
Build a BVH from a point cloud by creating a small AABB around each point.

```python
import numpy as np
from pytinybvh import BVH

points_np = np.random.rand(1000, 3).astype(np.float32)

# This creates AABBs and then builds the BVH.
bvh = BVH.from_points(points_np, radius=0.01)
```

---

### Ray Intersection

Once a BVH is built, you can perform fast intersection and occlusion queries.

#### Single Ray
```python
from pytinybvh import Ray

# Create a ray
ray = Ray(origin=[0.5, 0.2, -5.0], direction=[0, 0, 1])

# Intersect modifies the ray object in-place
bvh.intersect(ray)

# A miss is encoded with all the bits set to 1
miss = np.iinfo(np.uint32).max  # that is 4294967295 in unsigned integers, or -1 in signed integers

if ray.prim_id != miss: 
    print(f"Hit primitive {ray.prim_id} at distance t={ray.t:.3f}")
    print(f"Barycentric coords: u={ray.u:.3f}, v={ray.v:.3f}")
```

#### Batch of Rays
```python
# (N, 3) arrays for origins and directions
origins = np.array([[0.5, 0.2, -5], [10, 10, -5]], dtype=np.float32)
directions = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

# Returns a structured array with hit records
hits = bvh.intersect_batch(origins, directions)

print(hits['t'])
# prints:
# [(0, 5., 0.2, 0.3)]

print(hits['prim_id'].astype(np.int32))     # cast to signed integers to see the '-1' 
# prints:
# [(-1, inf, 0., 0.)]
```

You can also have a look at `tests.py` to see all the use cases.


## Running the demo viewer

I included a simple `visualise.py` script that opens a 3D viewer.

0. **Install dependencies:**
    ```bash
    uv pip install trimesh pyvista
    ```
   
1.  **Configure the script:**
    Open `test.py` and modify the configuration at the top:
    ```python
    TEST_SCENE_FILE = Path('dragon.ply') # File to load
    POINTS_ONLY = False                  # Set True for points, False for triangles
    VISUALIZE_DEPTH = 7                  # Max BVH depth to display
    ```

2.  **Run it:**
    ```bash
    python visualise.py
    ```

## Project Structure

```
pytinybvh/
├── deps/                   # C++ dependencies (submodules)
│   └── tinybvh/
├── src/
│   ├── pytinybvh.cpp       # C++ wrapper source
│   └── pytinybvh.pyi       # Python stub file
├── .gitignore
├── .gitattributes
├── .gitmodules             # Git submodule configuration
├── pyproject.toml          # Python build configuration
├── setup.py                # Build script for the C++ extension
├── tests.py                # Demo and tests script
├── visualise.py            # View the BVH with the geometry in 3D
├── LICENSE                 # MIT License
├── dragon.ply              # Stanford dragon as a.ply file
├── NOTICE.md
└── README.md
```


## Roadmap

The current version of `pytinybvh` does not yet provide all of `tinybvh`'s functionality.
I plan to expand the Python API to include everything. 

Immediate priorities:

- [x] Indexed Geometry Support
- [ ] Top-Level Acceleration Structure (TLAS) Support
- [x] Basic support for custom Geometry (via AABB builder)
- [ ] Proper support for Custom Geometry


## Acknowledgements

-   **Jacco Bikker** of course, for creating and open-sourcing the excellent `tinybvh` library

## Test Assets

The `dragon.ply` file included in this repository is a standard computer graphics test model provided for demonstration purposes.

-   **Model:** The Stanford Dragon
-   **Source:** [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3dscanrep/)
-   **Copyright:** The data is provided by the Stanford University Computer Graphics Laboratory. In line with their usage policy, I acknowledge them as the source of this data.