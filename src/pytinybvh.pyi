from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Optional, Union, List, Tuple, ClassVar
from enum import IntEnum

PathLike = Union[str, Path]
"""A type hint for file paths that can be a string or a pathlib.Path object."""

Vec3Like = Union[List[float], Tuple[float, float, float], np.ndarray]
"""
A type hint for objects that can be interpreted as a 3D vector,
including lists, tuples, and NumPy arrays of 3 floats.
"""

# A hint for the structured dtype of the BVH.nodes array.
# bvh_node_dtype = np.dtype([
#     ('aabb_min', ('<f4', (3,))),
#     ('left_first', '<u4'),
#     ('aabb_max', ('<f4', (3,))),
#     ('prim_count', '<u4')
# ])
bvh_node_dtype: np.dtype

# A hint for the structured dtype of the intersect_batch return array.
# hit_record_dtype = np.dtype([
#     ('prim_id', '<u4'),
#     ('inst_id', '<u4'),
#     ('t', '<f4'),
#     ('u', '<f4'),
#     ('v', '<f4')
# ])
hit_record_dtype: np.dtype

# A hint for the structured dtype of the TLAS instances array.
# instance_dtype = np.dtype([
#     ('transform', '<f4', (4, 4)),
#     ('blas_id', '<u4'),
#     ('mask', '<u4')
# ])
instance_dtype: np.dtype


class BuildQuality(IntEnum):
    """Enum for selecting BVH build quality."""

    Quick: ClassVar[BuildQuality]
    """Fastest build, lower quality queries."""
    Balanced: ClassVar[BuildQuality]
    """Balanced build time and query performance (default)."""
    High: ClassVar[BuildQuality]
    """Slowest build (uses spatial splits), highest quality queries."""

    def __int__(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class Ray:
    """Represents a ray for intersection queries."""
    origin: Vec3Like
    direction: Vec3Like
    t: float
    mask: int

    @property
    def u(self) -> float:
        """
        Barycentric u-coordinate for triangle hits, or the first
        texture coordinate for custom geometry like spheres and AABBs.
        """
        ...

    @property
    def v(self) -> float:
        """
        Barycentric v-coordinate for triangle hits, or the second
        texture coordinate for custom geometry like spheres and AABBs.
        """
        ...

    @property
    def prim_id(self) -> int:
        """The ID of the primitive that was hit. -1 if no hit."""
        ...

    @property
    def inst_id(self) -> int:
        """The ID of the instance that was hit. -1 if no hit."""
        ...

    def __init__(self, origin: Vec3Like, direction: Vec3Like, t: float = 1e30, mask: int = 0xFFFF) -> None: ...


class BVH:
    """A Bounding Volume Hierarchy for fast ray intersections."""

    @staticmethod
    def from_triangles(triangles: np.ndarray, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH from a standard triangle array. This is a convenience method that
        copies and reformats the data into the layout required by the BVH.

        Args:
            triangles (numpy.ndarray): A float32 array of shape (N, 3, 3) or (N, 9)
                                       representing N triangles.
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_points(points: np.ndarray, radius: float = 1e-05, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH from a point cloud. This is a convenience method that creates an
        axis-aligned bounding box for each point and builds the BVH from those.

        Args:
            points (numpy.ndarray): A float32 array of shape (N, 3) representing N points.
            radius (float): The radius used to create an AABB for each point.
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_vertices(vertices: np.ndarray, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH from a flat array of vertices in tinybvh's native format.

        This is a zero-copy operation. The BVH will hold a reference to the
        provided numpy array's memory buffer. The array must not be garbage-collected
        while the BVH is in use. The number of vertices must be a multiple of 3.

        Args:
            vertices (numpy.ndarray): A float32, C-contiguous array of shape (N * 3, 4).
                                      The 4th component is for padding and is ignored.
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_indexed_mesh(vertices: np.ndarray, indices: np.ndarray, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH from a vertex buffer and an index buffer.

        This is the most memory-efficient method for triangle meshes and allows for
        efficient refitting after vertex deformation. This is a zero-copy operation.
        The BVH will hold a reference to both provided numpy arrays.

        Args:
            vertices (numpy.ndarray): A float32, C-contiguous array of shape (V, 4), where V is the
                                      number of unique vertices. The 4th component is for padding.
            indices (numpy.ndarray): A uint32, C-contiguous array of shape (N, 3), where N is the
                                     number of triangles.
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_aabbs(aabbs: np.ndarray, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH from an array of Axis-Aligned Bounding Boxes.

        This is a zero-copy operation. The BVH will hold a reference to the
        provided numpy array's memory buffer. This is useful for building a BVH over
        custom geometry or for creating a Top-Level Acceleration Structure (TLAS).

        Args:
            aabbs (numpy.ndarray): A float32, C-contiguous array of shape (N, 2, 3),
                                   where each item is a pair of [min_corner, max_corner].
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def build_tlas(instances: np.ndarray, BLASes: List[BVH]) -> BVH:
        """
        Builds a Top-Level Acceleration Structure (TLAS) from a list of BVH instances.

        Args:
            instances (numpy.ndarray): A structured array with `instance_dtype` describing
                                       each instance's transform, blas_id, and mask.
            BLASes (List[BVH]): A list of the BVH objects to be instanced. The `blas_id`
                                in the instances array corresponds to the index in this list.

        Returns:
            BVH: A new BVH instance representing the TLAS.
        """
        ...

    @staticmethod
    def load(filepath: PathLike, vertices: np.ndarray, indices: Optional[np.ndarray] = None) -> BVH:
        """
        Loads a BVH from a file, re-linking it to the provided geometry.

        The geometry must be provided in the same layout as when the BVH was originally
        built and saved. If it was built from an indexed mesh, both the vertices and the indices must be provided.

        Args:
            filepath (str or pathlib.Path): The path to the saved BVH file.
            vertices (numpy.ndarray): A float32, C-style contiguous numpy array of shape (V, 4)
                                      representing the vertex data.
            indices (numpy.ndarray, optional): A uint32, C-style contiguous array of shape (N, 3)
                                               if the BVH was built from an indexed mesh.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    def save(self, filepath: PathLike) -> None:
        """
        Saves the BVH to a file.

        Args:
            filepath (str or pathlib.Path): The path where the BVH file will be saved.
        """
        ...

    def intersect(self, ray: Ray) -> float:
        """
        Performs an intersection query with a single ray.

        This method modifies the passed Ray object in-place if a closer hit is found.

        Args:
            ray (Ray): The ray to test. Its `t`, `u`, `v`, and `prim_id` attributes
                       will be updated upon a successful hit.

        Returns:
            float: The hit distance `t` if a hit was found, otherwise `infinity`.
        """
        ...

    def intersect_batch(self, origins: np.ndarray, directions: np.ndarray,
                        t_max: Optional[np.ndarray] = None, masks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs intersection queries for a batch of rays.

        This method leverages both multi-core processing (via OpenMP) and SIMD instructions
        (via tinybvh's Intersect256Rays functions) for maximum throughput on standard
        triangle meshes. For custom geometry like AABBs or spheres, it falls back to a
        serial implementation.

        Args:
            origins (numpy.ndarray): A (N, 3) float array of ray origins.
            directions (numpy.ndarray): A (N, 3) float array of ray directions.
            t_max (numpy.ndarray, optional): A (N,) float array of maximum intersection distances.
            masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                             For a ray to test an instance for intersection, the bitwise
                                             AND of the ray's mask and the instance's mask must be non-zero.
                                             If not provided, rays default to mask 0xFFFF (intersect all instances).

        Returns:
            numpy.ndarray: A structured array of shape (N,) with dtype
                [('prim_id', '<u4'), ('inst_id', '<u4'), ('t', '<f4'), ('u', '<f4'), ('v', '<f4')].
                For misses, prim_id and inst_id are -1 and t is infinity.
                For TLAS hits, inst_id is the instance index and prim_id is the primitive
                index within that instance's BLAS.
        """
        ...

    def is_occluded(self, ray: Ray) -> bool:
        """
        Performs an occlusion query with a single ray.

        Checks if any geometry is hit by the ray within the distance specified by `ray.t`.
        This is typically faster than `intersect` as it can stop at the first hit.

        Args:
            ray (Ray): The ray to test.

        Returns:
            bool: True if the ray is occluded, False otherwise.
        """
        ...

    def is_occluded_batch(self, origins: np.ndarray, directions: np.ndarray,
                          t_max: Optional[np.ndarray] = None, masks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs occlusion queries for a batch of rays, parallelized for performance.

        This method leverages multi-core processing (via OpenMP) for maximum throughput.

        Args:
            origins (numpy.ndarray): A (N, 3) float array of ray origins.
            directions (numpy.ndarray): A (N, 3) float array of ray directions.
            t_max (numpy.ndarray, optional): A (N,) float array of maximum occlusion distances.
                                             If a hit is found beyond this distance, it is ignored.
            masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                             For a ray to test an instance for intersection, the bitwise
                                             AND of the ray's mask and the instance's mask must be non-zero.
                                             If not provided, rays default to mask 0xFFFF (intersect all instances).

        Returns:
            numpy.ndarray: A boolean array of shape (N,) where `True` indicates occlusion.
        """
        ...

    def intersect_sphere(self, centre: Vec3Like, radius: float) -> bool:
        """
        Checks if any geometry intersects with a given sphere.

        This is useful for proximity queries or collision detection. It stops at the
        first intersection found. Note: This method is not implemented for custom
        geometry (AABBs, points) and will only work on triangle meshes.

        Args:
            center (Vec3Like): The center of the sphere.
            radius (float): The radius of the sphere.

        Returns:
            bool: True if an intersection is found, False otherwise.
        """
        ...

    def refit(self) -> None:
        """
        Refits the BVH to the current state of the source geometry, which is much
        faster than a full rebuild.

        Should be called after the underlying vertex data (numpy array used for construction)
        has been modified.

        Note: This will fail if the BVH was built with spatial splits (high-quality preset).
        """
        ...

    def optimize(self) -> None:
        """
        Optimizes the BVH tree structure to improve query performance.

        This is a costly operation best suited for static scenes. It works by
        re-inserting subtrees into better locations based on the SAH cost.

        Args:
            iterations (int): The number of optimization passes.
            extreme (bool): If true, a larger portion of the tree is considered
                            for optimization in each pass.
            stochastic (bool): If true, uses a randomized approach to select
                               nodes for re-insertion.
        """
        ...

    def compact(self) -> None:
        """
        Removes unused nodes from the BVH structure, reducing memory usage.

        This is useful after building with high quality (which may create
        spatial splits and more primitives) or after optimization, as these
        processes can leave gaps in the node array.
        """
        ...

    @property
    def nodes(self) -> np.ndarray:
        """The structured numpy array of BVH nodes."""
        ...

    @property
    def prim_indices(self) -> np.ndarray:
        """The array of primitive indices, ordered for locality."""
        ...

    @property
    def node_count(self) -> int:
        """Total number of nodes in the BVH."""
        ...

    @property
    def prim_count(self) -> int:
        """Total number of primitives in the BVH."""
        ...

    @property
    def aabb_min(self) -> np.ndarray:
        """The minimum corner of the root axis-aligned bounding box."""
        ...

    @property
    def aabb_max(self) -> np.ndarray:
        """The maximum corner of the root axis-aligned bounding box."""
        ...

    @property
    def quality(self) -> BuildQuality:
        """The build quality level used to construct the BVH."""
        ...

    @property
    def sah_cost(self) -> float:
        """Calculates the Surface Area Heuristic (SAH) cost of the BVH tree."""
        ...