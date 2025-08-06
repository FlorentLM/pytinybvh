from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Optional, Union, List, Tuple
from enum import Enum


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
#     ('t', '<f4'),
#     ('u', '<f4'),
#     ('v', '<f4')
# ])
hit_record_dtype: np.dtype


class BuildQuality(Enum):
    """Enum for selecting BVH build quality."""
    Quick: int
    Balanced: int
    High: int


class Ray:
    """Represents a ray for intersection queries."""
    origin: Vec3Like
    direction: Vec3Like
    t: float

    @property
    def u(self) -> float:
        """Barycentric u-coordinate of the hit."""
        ...

    @property
    def v(self) -> float:
        """Barycentric v-coordinate of the hit."""
        ...

    @property
    def prim_id(self) -> int:
        """The ID of the primitive that was hit. -1 if no hit."""
        ...

    def __init__(self, origin: Vec3Like, direction: Vec3Like, t: float = 1e30) -> None: ...


class BVH:
    """A Bounding Volume Hierarchy for fast ray intersections."""

    @staticmethod
    def from_triangles(triangles: np.ndarray, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH for triangles from a (N, 3, 3) or (N, 9) float array.

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
        Builds a BVH for points from a (N, 3) float array.

        Internally, this represents each point as a small axis-aligned bounding box.

        Args:
            points (numpy.ndarray): A float32 array of shape (N, 3) representing N points.
            radius (float): The radius used to create an AABB for each point.
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_vertices_4f(vertices: np.ndarray, quality: BuildQuality = ...) -> BVH:
        """
        Builds a BVH from a pre-formatted (M, 4) vertex array.

        This is a zero-copy operation. The BVH will hold a reference to the
        provided numpy array's memory buffer. The array must not be garbage-collected
        while the BVH is in use. M must be a multiple of 3.

        Args:
            vertices (numpy.ndarray): A float32 array of shape (M, 4).
            quality (BuildQuality): The desired quality of the BVH.

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def load(vertices: np.ndarray, filepath: PathLike) -> BVH:
        """
        Loads a BVH from a file, requires the original vertex data.

        Per design tinybvh does not save the geometric vertex data in its file format;
        only the acceleration structure itself (the nodes and primitive indices).

        Thus, the vertex data that was used to create the original BVH must be provided:
        this function re-links the loaded acceleration structure to the vertex data in memory.

        Args:
            vertices (numpy.ndarray): A float32, C-style contiguous numpy array of shape (M, 4) representing
                the vertex data. This must be the same data that was used when the BVH was originally built and saved.
            filepath (str or pathlib.Path): The path to the saved BVH file.

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

    def intersect_batch(self, origins: np.ndarray, directions: np.ndarray, t_max: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs intersection queries for a batch of rays.

        Args:
            origins (numpy.ndarray): A (N, 3) float array of ray origins.
            directions (numpy.ndarray): A (N, 3) float array of ray directions.
            t_max (numpy.ndarray, optional): A (N,) float array of maximum intersection distances.

        Returns:
            numpy.ndarray: A structured array of shape (N,) with dtype
                           [('prim_id', '<u4'), ('t', '<f4'), ('u', '<f4'), ('v', '<f4')].
                           For misses, prim_id is -1 and t is infinity.
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

    def is_occluded_batch(self, origins: np.ndarray, directions: np.ndarray, t_max: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs occlusion queries for a batch of rays.

        Args:
            origins (numpy.ndarray): A (N, 3) float array of ray origins.
            directions (numpy.ndarray): A (N, 3) float array of ray directions.
            t_max (numpy.ndarray, optional): A (N,) float array of maximum occlusion distances.
                                             If a hit is found beyond this distance, it is ignored.

        Returns:
            numpy.ndarray: A boolean array of shape (N,) where `True` indicates occlusion.
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