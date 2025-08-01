# pytinybvh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple Python bindings for the great header-only C++ Bounding Volume Hierarchy (BVH) library [tinybvh](https://github.com/jbikker/tinybvh) by Jacco Bikker.

Expose `tinybvh`'s fast BVH construction algorithms to Python, to use for CPU-side BVH generation in Python applications.
Typically used for real-time ray tracing with PyOpenGL or Vulkan, collision detection, and other spatial queries.

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

4.  **Install and Compile:**
    Run the installation in editable mode. This will automatically download Python dependencies and compile the C++ extension module.
    ```bash
    uv pip install -e .
    ```

If the process completes without errors, the `pytinybvh` module is now installed and ready to be used in your virtual environment.

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

## Acknowledgements

-   **Jacco Bikker** of course, for creating and open-sourcing the excellent `tinybvh` library

## Test Assets

The `dragon.ply` file included in this repository is a standard computer graphics test model provided for demonstration purposes.

-   **Model:** The Stanford Dragon
-   **Source:** [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3dscanrep/)
-   **Copyright:** The data is provided by the Stanford University Computer Graphics Laboratory. In line with their usage policy, I acknowledge them as the source of this data.