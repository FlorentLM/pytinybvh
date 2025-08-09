# pytinybvh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python bindings for the great C++ Bounding Volume Hierarchy (BVH) library [tinybvh](https://github.com/jbikker/tinybvh) by Jacco Bikker.

Exposes `tinybvh`'s fast BVH construction algorithms to Python.
For example, the BVH can be used as-is in a SSBO for real time ray-tracing with PyOpenGL or Vulkan, or for CPU-side computations like collision detection, etc.

**Note:** Not all functionalities are implemented yet!

<div style="text-align:center">
<img alt="Screenshot of a test model in a Bounding Volume Hierarchy" src="img/screenshot.png" title="Screenshot" width="500"/>
</div>

## Prerequisites

Before installing, you need a few tools on your system:

1.  **A C++20 Compatible Compiler:**
    -   **Windows:** MSVC (Visual Studio 2019 or newer with the "Desktop development with C++" workload)
    -   **Linux:** GCC 10+ or Clang 11+
    -   **macOS:** Xcode with modern Clang
2.  **Python 3.9+**
3.  **Git:** Required for cloning the repository and its dependencies

## Installation

The C++ dependency (`tinybvh`) is included as a Git submodule.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FlorentLM/pytinybvh.git
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

4.   **Compile and install:**
    This command will automatically download Python dependencies (if any) and compile the C++ extension module.

    ```bash
    uv pip install .
    ```
    This installs the core `pytinybvh` library. To include packages for testing and visualisation, you can install "extras":
    ```bash
    # Install everything needed for development (testing, visualisation)
    uv pip install .[dev]
    
    # Or install specific groups
    uv pip install .[test]
    uv pip install .[visualise]
    ```

If the process completes without errors, the `pytinybvh` module is now installed and ready to be used in your virtual environment.

5. **Use in other projects:**
    From the virtual environment of your other project, run the installation of `pytinybvh` in editable mode.
    ```bash
    uv pip install -e /path/to/this/repo
    ```

## Performance & Build optimizations

For users building from source who want maximum performance on their specific machine, `pytinybvh` supports two key optimizations that can be enabled during installation.

### 1. Parallelization with OpenMP (Enabled by default if available)

The batch intersection and occlusion methods (`intersect_batch`, `is_occluded_batch`) are parallelized using OpenMP to take advantage of multiple CPU cores. The setup script should automatically detect if your compiler supports OpenMP and enable it.

-   **GCC/Clang:** Requires `-fopenmp`.
-   **MSVC:** Requires `/openmp`.

If your compiler does not support OpenMP, `pytinybvh` will compile and run in serial (single-threaded) mode. You will see a message during installation indicating whether parallelization is enabled.

### 2. AVX and AVX2 SIMD Optimizations

`tinybvh` also includes highly optimized code paths that use AVX2 and FMA instructions for significant speedups in BVH traversal on CPU.
#### AVX Support (Enabled by default if available)

Modern CPUs (since ~2011) support AVX. This instruction set is required for `tinybvh`'s highly optimized `Intersect256RaysSSE` function, which provides a significant speedup for `intersect_batch`. The build script will **automatically detect and enable AVX support** if your compiler and system can handle it.

#### AVX2 Support (Opt-in)

Even faster code paths are available using AVX2 and FMA instructions (found on Intel Haswell CPUs from ~2013 and newer, and AMD Zen from ~2017 and newer). However, binaries compiled with these flags _will_ crash on CPUs that do not support them. For this reason, AVX2 is an opt-in feature only.

If you are compiling on a machine with a modern CPU (Intel Haswell or newer, AMD Zen or newer) and intend to run the code on the same or a similarly capable machine, you can enable AVX2 support with an environment variable before running the installation.

**On Linux/macOS:**

```bash
PYTINYBVH_ENABLE_AVX2=1 uv pip install .
# or if using pip directly:
# PYTINYBVH_ENABLE_AVX2=1 pip install .
```

**On Windows (Command Prompt):**
```cmd
set PYTINYBVH_ENABLE_AVX2=1
uv pip install .
```

**On Windows (PowerShell)**
```powershell
$env:PYTINYBVH_ENABLE_AVX2="1"
uv pip install .
```

The build script will print a message confirming that AVX2 optimizations are being compiled. If you enable this option, the resulting wheel will _only_ be usable on machines that support AVX2.

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
Builds a BVH from a list of pre-defined Axis-Aligned Bounding Boxes. Useful for custom geometry or for building a Top-Level Acceleration Structure (TLAS).

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

**Note:** When using a BVH built from AABBs, intersection tests are performed directly against the bounding boxes.
The hit record will contain the primitive ID of the AABB, and the `u` and `v` coordinates will contain 2D position of the hit on the specific face of the box that was intersected.

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
Builds a BVH from a point cloud.

It works by first creating an axis-aligned bounding box for each point to build the hierarchy, then uses ray-sphere tests inside the leaf nodes.

```python
import numpy as np
from pytinybvh import BVH

points_np = np.random.rand(1000, 3).astype(np.float32)

# This creates AABBs and then builds the BVH.
bvh = BVH.from_points(points_np, radius=0.01)

```

**Note:** Since the primitives are spheres, the `u` and `v` coordinates represent the spherical coordinates (lat, lon) of the intersection point.

---

### Ray Intersection

Once a BVH is built, you can perform fast intersection and occlusion queries.

#### Single Ray
```python
from pytinybvh import Ray

# Create a ray
ray = Ray(origin=[0.5, 0.2, -5.0], direction=[0, 0, 1])

# Intersect modifies the ray object in-place. Useful for multiple bounces.
bvh.intersect(ray)

if ray.prim_id != -1: 
    print(f"Hit primitive {ray.prim_id} at distance t={ray.t:.3f}")
    print(f"Barycentric coords: u={ray.u:.3f}, v={ray.v:.3f}")
```

#### Batch of Rays
```python
      
# (N, 3) arrays for origins and directions
origins = np.array([[0.5, 0.2, -5], [10, 10, -5]], dtype=np.float32)
directions = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

# Returns a structured numpy array with hit records
hits = bvh.intersect_batch(origins, directions)

# You can access columns like a dictionary
hit_distances = hits['t']
primitive_ids = hits['prim_id'].astype(np.int32) # Cast to signed int to see -1 for misses

print("Batch Hit Results:")
for i in range(len(hits)):
    if primitive_ids[i] != -1:
        print(f"  Ray {i} hit primitive {primitive_ids[i]} at t={hits[i]['t']:.3f}")
    else:
        print(f"  Ray {i} missed.")

# prints:
# Batch Hit Results:
#   Ray 0 hit primitive 0 at t=5.000
#   Ray 1 missed.

```

## Running Tests

The test suite uses `pytest`.

1.  **Install test dependencies:**
    ```bash
    # Install with the [test] extra if you haven't already
    uv pip install .[test]
    ```
   
2.  **Run the test suite:**
    ```bash
    pytest
    ```
    The tests include a visualisation of the test scene which can be run by executing `test_pytinybvh.py` directly (`python test_pytinybvh.py`). This requires the `visualise` dependencies.

## Running the demo viewer

I also included a simple `visualise.py` script that opens a 3D viewer.

1.  **Install visualisation dependencies:**
    ```bash
    # Install with the [visualise] extra if you haven't already
    uv pip install .[visualise]
    ```
   
2.  **Configure the script:**
    Open `visualise.py` and modify the configuration at the top:
    ```python
    TEST_SCENE_FILE = Path('sneks.ply')  # File to load
    POINTS_ONLY = False                  # Set True for points, False for triangles
    VISUALISE_DEPTH = 7                  # Max BVH depth to display
    ```

3.  **Run it:**
    ```bash
    python visualise.py
    ```

## Project Structure

```
pytinybvh/
├── assets/                 # Test models, etc
│   └── sneks.ply
├── deps/                   # C++ dependencies (submodules)
│   └── tinybvh/
├── img/                    # Images used in this README
│   └── screenshot.png
├── src/
│   ├── pytinybvh.cpp       # C++ wrapper source
│   └── pytinybvh.pyi       # Python stub file
├── .gitignore
├── .gitattributes
├── .gitmodules             # Git submodule configuration
├── pyproject.toml          # Python build configuration
├── setup.py                # Build script for the C++ extension
├── test_pytinybvh.py       # Demo and tests
├── visualise.py            # View the BVH with the geometry in 3D
├── LICENSE                 # MIT License
├── NOTICE.md
└── README.md
```


## Roadmap

The current version of `pytinybvh` does not yet provide all of `tinybvh`'s functionality.
I plan to expand the Python API to include everything. 

Immediate priorities:

- [x] Indexed Geometry Support
- [x] Custom Geometry Primitives (AABBs and Spheres)
- [x] Top-Level Acceleration Structure (TLAS) Support for instancing
- [ ] Support for more `tinybvh` build presets and layouts (BVH8, etc)

## Remarks

### A Note on performance and concurrency (Update!)

-   **Multi-Core processing:** The batch intersection methods (`intersect_batch` and `is_occluded_batch`) are now fully parallelized for _all_ supported geometry types: triangles, custom AABBs, and custom spheres.
-   **SIMD optimizations:** This is still only for standard triangle meshes, but such case `intersect_batch` gains an additional speedup by using AVX SIMD instructions to process rays in large packets, if your CPU and build configuration support it.

## Acknowledgements

-   **Jacco Bikker** of course, for creating and open-sourcing the excellent `tinybvh` library

## Test Assets

The `sneks` model included in this repository is an original model. Feel free to reuse it. :)

-   **Source:** [here](https://github.com/FlorentLM/pytinybvh/blob/main/assets/sneks.ply)
-   **Size:** 9.37 Mb
-   **Vertices:** 169 678
-   **Triangles:** 338 120