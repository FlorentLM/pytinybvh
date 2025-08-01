# pytinybvh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple Python bindings for the great C++ Bounding Volume Hierarchy (BVH) library [tinybvh](https://github.com/jbikker/tinybvh) by Jacco Bikker.

Exposes `tinybvh`'s fast BVH construction algorithms to Python, to use for CPU-side BVH generation in Python applications.
Typically used for real-time ray tracing with PyOpenGL or Vulkan, collision detection, etc.

**Note:** For now, `pytinybvh` only provides the core functionality for building a BVH with the SAH method, from triangle or point data
(see below for Roadmap).

The output is provided as raw numpy arrays, designed to be uploaded directly to the GPU as Shader Storage Buffer Objects (SSBOs).

## Prerequisites

Before installing, you need a few tools on your system:

1.  **A C++20 Compatible Compiler:**
    -   **Windows:** MSVC (Visual Studio 2019 or newer with the "Desktop development with C++" workload)
    -   **Linux:** GCC 10+ or Clang 11+
    -   **macOS:** Xcode with modern Clang
2.  **Python 3.12+**
3.  **Git:** Required for cloning the repository and its C++ dependencies (submodules)

## Installation

The C++ dependencies (`tinybvh` and `glm`) are included as Git submodules.

1.  **Clone the repository:**
    ```bash
    git clone https://FlorentLM/pytinybvh.git
    cd pytinybvh
    ```

2.  **Initialize the C++ Submodules:**
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

If the process completes without errors, the `pytinybvh` module is now installed and ready to be used in your virtual environment. This will automatically download Python dependencies and compile the C++ extension module.


5. **Use in other projects:**
    From the virtual environment of your other project, run the installation of `pytinybvh` in editable mode.
    ```bash
   # Still with uv
    uv pip install -e /path/to/this/repo
    ```
    
## Usage

The module provides two functions that take a numpy array and return a tuple of two numpy arrays for the BVH nodes and the primitive indices.

### For Triangle Meshes

The input must be a NumPy array of shape `(N, 9)`, where N is the number of triangles.

```python
import numpy as np
import pytinybvh

# Create a flat array of triangle data (2 triangles)
triangles_np = np.array([
    # v0.xyz,      v1.xyz,      v2.xyz
    [0, 0, 0,     1, 0, 0,     0, 1, 0],
    [2, 2, 2,     3, 2, 2,     2, 3, 2],
], dtype=np.float32)

# Build the BVH
bvh_nodes, prim_indices = pytinybvh.from_triangles(triangles_np)

# For rendering, reorder the original triangles
reordered_triangles = triangles_np[prim_indices]

# 'bvh_nodes' and 'reordered_triangles' can now be uploaded to SSBOs
```

### For Point Clouds

The input must be a NumPy array of shape `(N, 3)`, where N is the number of points.

```python
import numpy as np
import pytinybvh

# Create an array of 3D points
points_np = np.array([
    [0, 0, 0],
    [1, 2, 3],
    [4, 5, 6],
    # ... more points
], dtype=np.float32)

# Build the BVH
bvh_nodes, prim_indices = pytinybvh.from_points(points_np)

# For rendering, reorder the original points
reordered_points = points_np[prim_indices]

# 'bvh_nodes' and 'reordered_points' can now be uploaded to SSBOs
```

## Running the Demo script

I included a simple `test.py` script that includes a 3D viewer.

0. **Install dependencies:**
    ```bash
    uv pip install trimesh pyvista
    ```
   
1.  **Configure the script:**
    Open `test.py` and modify the configuration flags at the top:
    ```python
    TEST_SCENE_FILE = Path('dragon.ply') # File to load
    POINTS_ONLY = False                  # Set True for points, False for triangles
    VISUALIZE = True                     # Set True to open the 3D viewer (needs pyvista)
    VISUALIZE_DEPTH = 7                  # Max BVH depth to display
    ```

2.  **Run it:**
    ```bash
    python test.py
    ```

## Project Structure

```
pytinybvh/
├── deps/                   # C++ dependencies (submodules)
│   ├── glm/
│   └── tinybvh/
├── src/                    # C++ wrapper source
│   └── pytinybvh.cpp
├── .gitmodules             # Git submodule configuration
├── pyproject.toml          # Python build configuration
├── setup.py                # Build script for the C++ extension
├── test.py                 # Demo and visualization script
├── LICENSE                 # MIT License
├── dragon.ply              # Stanford dragon as a.ply file
└── README.md
```


## Roadmap

The current version of `pytinybvh` provides only the core functionality for building a high-quality BVH from triangle or point data.
However, `tinybvh` is much more powerful and I plan to expand the Python API to include everything. 

Contributions are welcome :)

### Core BVH Features

*   **1. Expose different build qualities:**
    -   **What:** Wrap the `BuildQuick` and `BuildHQ` (SBVH) methods.
    -   **Why:** Allows a trade-off between build time and traversal performance. A potential API could be `pytinybvh.build(..., quality='fast'|'default'|'high')`
        -   `fast`: For dynamic scenes where build speed is critical.
        -   `default`: The current high-quality SAH builder.
        -   `high`: For static scenes where maximum ray tracing performance is desired, at the cost of a slower initial build.

*   **2. BVH Refitting (`Refit`):**
    -   **What:** Expose the `Refit()` method.
    -   **Why:** This is a major performance optimization for dynamic scenes where vertices move but the topology is constant (e.g., skinned character animation, cloth simulation). Refitting only updates the AABB positions and is an order of magnitude faster than a full rebuild.

*   **3. Indexed Geometry Support:**
    -   **What:** Add a build function that accepts a vertex array `(V, 3)` and a face index array `(F, 3)`.
    -   **Why:** This is far more memory-efficient for standard meshes, as vertex data is not duplicated for each triangle.

*   **4. BVH Caching (`Save` / `Load`):**
    -   **What:** Wrap the `Save()` and `Load()` methods.
    -   **Why:** For very large static scenes, building the BVH can take time. This would allow building the BVH once, save it to a file, and load it almost instantly in future runs.
    -   **NOTE**: Probably could be directly implemented in Python, it's fast enough
    
### Advanced features

*   **5. Top-Level Acceleration Structure (TLAS) Support:**
    -   **What:** Expose the `BLASInstance` class and the `Build` overload for instances.
    -   **Why:** This is essential for rendering large, complex scenes composed of many distinct objects, especially if they are moving (e.g., cars in a city, trees in a forest). It allows building a two-level BVH (a BVH of BVHs), which is the standard approach in modern ray tracers.

*   **6. CPU-side Ray Queries:**
    -   **What:** Wrap the `Intersect` and `IsOccluded` methods.
    -   **Why:** While the primary goal is GPU rendering, exposing CPU-side queries would be cool for Python-only applications like:
        -   Collision detection
        -   CPU-based ray casting for mouse picking or physics
        -   Debugging and verification

*   **7. Support for Custom Geometry:**
    -   **What:** Expose the `Build` overload that accepts a callback function for getting custom primitive AABBs.
    -   **Why:** This would allow building a BVH over arbitrary Python objects, not just triangles or points, by providing a function that returns the bounding box for a given object index.


## Acknowledgements

-   **Jacco Bikker** of course, for creating and open-sourcing the excellent `tinybvh` library

## Test Assets

The `dragon.ply` file included in this repository is a standard computer graphics test model provided for demonstration purposes.

-   **Model:** The Stanford Dragon
-   **Source:** [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3dscanrep/)
-   **Copyright:** The data is provided by the Stanford University Computer Graphics Laboratory. In line with their usage policy, I acknowledge them as the source of this data.