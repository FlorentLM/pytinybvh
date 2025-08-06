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

### For Triangle Meshes

The input must be a NumPy array of shape `(N, 9)` or `(N, 3, 3)`, where N is the number of triangles.

```python
import numpy as np
from pytinybvh import BVH

# Create a flat array of triangle data (2 triangles)
triangles_np = np.array([
    # v0.xyz,      v1.xyz,      v2.xyz
    [0, 0, 0,     1, 0, 0,     0, 1, 0],
    [2, 2, 2,     3, 2, 2,     2, 3, 2],
], dtype=np.float32)

# Build the BVH
bvh = BVH.from_points(triangles_np)

# For rendering, reorder the original triangles
reordered_triangles = triangles_np[bvh.prim_indices]

# 'bvh.nodes' and 'reordered_triangles' can now be uploaded to SSBOs
```

### For Point Clouds

The input must be a NumPy array of shape `(N, 3)`, where N is the number of points.

```python
import numpy as np
from pytinybvh import BVH

# Create an array of 3D points
points_np = np.array([
    [0, 0, 0],
    [1, 2, 3],
    [4, 5, 6],
    # ... more points
], dtype=np.float32)

# Build the BVH
bvh = BVH.from_points(points_np)

# For rendering, reorder the original points
reordered_points = points_np[bvh.prim_indices]

# 'bvh.nodes' and 'reordered_points' can now be uploaded to SSBOs
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

- [ ] Indexed Geometry Support
- [ ] Top-Level Acceleration Structure (TLAS) Support
- [ ] Proper support for Custom Geometry (probably replacing the current `from_points()` method)


## Acknowledgements

-   **Jacco Bikker** of course, for creating and open-sourcing the excellent `tinybvh` library

## Test Assets

The `dragon.ply` file included in this repository is a standard computer graphics test model provided for demonstration purposes.

-   **Model:** The Stanford Dragon
-   **Source:** [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3dscanrep/)
-   **Copyright:** The data is provided by the Stanford University Computer Graphics Laboratory. In line with their usage policy, I acknowledge them as the source of this data.