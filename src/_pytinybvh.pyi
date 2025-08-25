from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Optional, Union, List, Tuple, ClassVar, Dict, Any
from enum import IntEnum


# Type aliases

PathLike = Union[str, Path]
"""
A type hint for file paths that can be a string or a pathlib.Path object.
"""

Vec3Like = Union[List[float], Tuple[float, float, float], np.ndarray]
"""
A type hint for objects that can be interpreted as a 3D vector,
including lists, tuples, and numpy arrays of 3 floats.
"""

# numpy dtypes

bvh_node_dtype: np.dtype
"""
A hint for the structured dtype of the BVH.nodes array.

bvh_node_dtype = np.dtype([
    ('aabb_min', ('<f4', (3,))),
    ('left_first', '<u4'),
    ('aabb_max', ('<f4', (3,))),
    ('prim_count', '<u4')
])
"""

hit_record_dtype: np.dtype
"""
A hint for the structured dtype of the intersect_batch return array.

hit_record_dtype = np.dtype([
    ('prim_id', '<u4'),
    ('inst_id', '<u4'),
    ('t', '<f4'),
    ('u', '<f4'),
    ('v', '<f4')
])
"""

instance_dtype: np.dtype
"""
A hint for the structured dtype of the TLAS instances array.

instance_dtype = np.dtype([
    ('transform', '<f4', (4, 4)),
    ('blas_id', '<u4'),
    ('mask', '<u4')
])
"""


# Top-level functions

def hardware_info() -> Dict[str, Any]:
    """
    Returns a dictionary detailing the compile-time and runtime capabilities of the library.

    This includes detected SIMD instruction sets and which BVH layouts support
    conversion and traversal on the current system.

    Returns:
        Dict[str, Any]: A dictionary with the hardware info.
    """
    ...

def supports_layout(layout: Layout, for_traversal: bool = True) -> bool:
    """
    Checks if the current system supports a given BVH layout.

    Args:
        layout (Layout): The layout to check.
        for_traversal (bool): If True (default), checks if the layout is supported for
                              ray traversal. If False, checks if it's supported for
                              conversion (which is always True for valid layouts).

    Returns:
        bool: True if the layout is supported, False otherwise.
    """
    ...

def require_layout(layout: Layout, for_traversal: bool = True) -> None:
    """
    Asserts that a given BVH layout is supported, raising a RuntimeError if not.

    This is useful for writing tests or code that depends on a specific high-performance
    layout being available.

    Args:
        layout (Layout): The layout to require.
        for_traversal (bool): If True (default), requires traversal support.

    Raises:
        RuntimeError: If the layout is not supported on the current system.
    """
    ...


# Enums

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


class GeometryType(IntEnum):
    """Enum for the underlying geometry type of the BVH."""

    Triangles: ClassVar[GeometryType]
    """The BVH was built over a triangle mesh."""
    AABBs: ClassVar[GeometryType]
    """The BVH was built over custom Axis-Aligned Bounding Boxes."""
    Spheres: ClassVar[GeometryType]
    """The BVH was built over a point cloud with a radius (spheres)."""

    def __int__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


class Layout(IntEnum):
    """Enum for the internal memory layout of the BVH."""

    Standard: ClassVar[Layout]
    """Standard BVH layout. Always available and traversable."""

    SoA: ClassVar[Layout]
    """Structure of Arrays layout, optimized for AVX/NEON traversal."""

    BVH_GPU: ClassVar[Layout]
    """Aila & Laine layout, optimized for GPU traversal (scalar traversal on CPU)."""

    MBVH4: ClassVar[Layout]
    """
    4-wide MBVH layout. This is a structural format used as an intermediate for
    other wide layouts. Not directly traversable.
    """

    MBVH8: ClassVar[Layout]
    """
    8-wide MBVH layout. This is a structural format used as an intermediate for
    other wide layouts. Not directly traversable.
    """

    BVH4_CPU: ClassVar[Layout]
    """4-wide BVH layout, optimized for SSE CPU traversal."""

    BVH4_GPU: ClassVar[Layout]
    """Quantized 4-wide BVH layout for GPUs (scalar traversal on CPU)."""

    CWBVH: ClassVar[Layout]
    """Compressed 8-wide BVH layout, optimized for AVX traversal."""

    BVH8_CPU: ClassVar[Layout]
    """8-wide BVH layout, optimized for AVX2 CPU traversal."""

    def __int__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


class CachePolicy(IntEnum):
    """Enum for managing cached BVH layouts after conversion."""

    ActiveOnly: ClassVar[CachePolicy]
    """
    (Default) Free memory of any non-active layouts after a conversion.
    This minimizes memory usage.
    """

    All: ClassVar[CachePolicy]
    """
    Keep all generated layouts in memory. This uses more memory but makes
    switching back to a previously used layout instantaneous.
    """

    def __int__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


class PacketMode(IntEnum):
    """Enum for controlling SIMD packet traversal in batched queries."""

    Auto: ClassVar[PacketMode]
    """Use packets for coherent rays with a shared origin."""

    Never: ClassVar[PacketMode]
    """Always use scalar traversal. Safest for non-coherent rays."""

    Force: ClassVar[PacketMode]
    """Force packet traversal. Unsafe for non-coherent rays."""

    def __int__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


# Main classes

class Ray:
    """Represents a ray for intersection queries."""

    origin: Vec3Like
    """The origin point of the ray (list, tuple, or numpy array)."""

    direction: Vec3Like
    """The direction vector of the ray (list, tuple, or numpy array)."""

    t: float
    """The maximum distance for intersection. Updated with hit distance."""

    mask: int
    """The visibility mask for the ray."""

    @property
    def u(self) -> float:
        """
        Barycentric u-coordinate, or texture coordinate u when using custom geometry.
        """
        ...

    @property
    def v(self) -> float:
        """
        Barycentric v-coordinate, or texture coordinate v when using custom geometry.
        """
        ...

    @property
    def prim_id(self) -> int:
        """The ID of the primitive hit (-1 for miss)."""
        ...

    @property
    def inst_id(self) -> int:
        """The ID of the instance hit (-1 for miss or BLAS hit)."""
        ...

    def __init__(self, origin: Vec3Like, direction: Vec3Like, t: float = 1e30, mask: int = 0xFFFF) -> None: ...


class BVH:
    """A Bounding Volume Hierarchy for fast ray intersections."""

    # Zero-copy builders

    @staticmethod
    def from_vertices(vertices: np.ndarray, quality: BuildQuality = ..., traversal_cost: float = ..., intersection_cost: float = ..., hq_bins : int = ...) -> BVH:
        """
        Builds a BVH from a flat array of vertices (N * 3, 4).

        This is a zero-copy operation. The BVH will hold a reference to the
        provided numpy array's memory buffer. The array must not be garbage-collected
        while the BVH is in use. The number of vertices must be a multiple of 3.

        Args:
            vertices (numpy.ndarray): A float32 array of shape (M, 4).
            quality (BuildQuality): The desired quality of the BVH.
            traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
            intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
            hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

        Returns:
            BVH: A new BVH instance.
        """
        ...


    @staticmethod
    def from_indexed_mesh(vertices: np.ndarray, indices: np.ndarray, quality: BuildQuality = ..., traversal_cost: float = ..., intersection_cost: float = ..., hq_bins : int = ...) -> BVH:
        """
        Builds a BVH from a vertex buffer and an index buffer.

        This is the most memory-efficient method for triangle meshes and allows for
        efficient refitting after vertex deformation. This is a zero-copy operation.
        The BVH will hold a reference to both provided numpy arrays.

        Args:
            vertices (numpy.ndarray): A float32 array of shape (V, 4), where V is the number of unique vertices.
            indices (numpy.ndarray): A uint32 array of shape (N, 3), where N is the number of triangles.
            quality (BuildQuality): The desired quality of the BVH.
            traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
            intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
            hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_aabbs(aabbs: np.ndarray, quality: BuildQuality = ..., traversal_cost: float = ..., intersection_cost: float = ..., hq_bins : int = ...) -> BVH:
        """
        Builds a BVH from an array of Axis-Aligned Bounding Boxes.

        This is a zero-copy operation. The BVH will hold a reference to the
        provided numpy array's memory buffer. This is useful for building a BVH over
        custom geometry or for creating a Top-Level Acceleration Structure (TLAS).

        Args:
            aabbs (numpy.ndarray): A float32, C-contiguous array of shape (N, 2, 3),
                                   where each item is a pair of [min_corner, max_corner].
            quality (BuildQuality): The desired quality of the BVH.
            traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
            intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
            hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

        Returns:
            BVH: A new BVH instance.
        """
        ...


    # Convenience builders

    @staticmethod
    def from_triangles(triangles: np.ndarray, quality: BuildQuality = ..., traversal_cost: float = ..., intersection_cost: float = ..., hq_bins : int = ...) -> BVH:
        """
        Builds a BVH from a standard triangle array. This is a convenience method that
        copies and reformats the data into the layout required by the BVH.

        Args:
            triangles (numpy.ndarray): A float32 array of shape (N, 3, 3) or (N, 9) representing N triangles.
            quality (BuildQuality): The desired quality of the BVH.
            traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
            intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
            hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

        Returns:
            BVH: A new BVH instance.
        """
        ...

    @staticmethod
    def from_points(points: np.ndarray, radius: float = 1e-05, quality: BuildQuality = ..., traversal_cost: float = ..., intersection_cost: float = ..., hq_bins : int = ...) -> BVH:
        """
        Builds a BVH from a point cloud. This is a convenience method that creates an
        axis-aligned bounding box for each point and builds the BVH from those.

        Args:
            points (numpy.ndarray): A float32 array of shape (N, 3) representing N points.
            radius (float): The radius used to create an AABB for each point.
            quality (BuildQuality): The desired quality of the BVH.
            traversal_cost (float, optional): The traversal cost for the SAH builder. Defaults to 1.
            intersection_cost (float, optional): The intersection cost for the SAH builder. Defaults to 1.
            hq_bins (int, optional): The number of bins to use for the high-quality build algorithm (SBVH).

        Returns:
            BVH: A new BVH instance.
        """
        ...

    # Cache policy management

    def set_cache_policy(self, policy: CachePolicy) -> None:
        """
        Sets caching policy for converted layouts.

        Args:
            policy (CachePolicy): The new policy to use (ActiveOnly or All).
        """
        ...

    def clear_cached_layouts(self) -> None:
        """
        Frees the memory of all cached layouts, except for the active one and the base layout.
        """
        ...

    # Conversion, TLAS building

    def convert_to(self, layout: Layout, compact: bool = True, strict: bool = False) -> None:
        """
        Converts the BVH to a different internal memory layout, modifying it in-place.

        This allows optimizing the BVH for different traversal algorithms (SSE, AVX, etc.).
        The caching policy of converted layouts can be controlled (see `set_cache_policy` and `clear_cached_layouts`).

        Args:
            layout (Layout): The target memory layout.
            compact (bool): Whether to compact the BVH during conversion. Defaults to True.
            strict (bool): If True, raises a RuntimeError if the target layout is not
                           supported for traversal on the current system. Defaults to False.
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

    # Intersection methods

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

    def intersect_batch(self,
                        origins: np.ndarray,
                        directions: np.ndarray,
                        t_max: Optional[np.ndarray] = None,
                        masks: Optional[np.ndarray] = None,
                        packet: PacketMode = PacketMode.Auto,
                        same_origin_eps: float = 1e-6,
                        max_spread: float = 1.0,
                        warn_on_incoherent: bool = True) -> np.ndarray:
        """
        Performs intersection queries for a batch of rays.

        This method is highly parallelized using multi-core processing for all geometry types
        (triangles, AABBs, spheres). For standard triangle meshes, it also leverages SIMD
        instructions where available for maximum throughput.

        .. warning::
            Packet traversal (`Auto` or `Force`) is highly optimized but
            makes strict assumptions about the input rays:
                - Shared origin: All rays in a batch must share the same origin point.
                - Coherent directions: The ray directions should form a coherent frustum,
                    like rays cast from a camera through pixels on a screen.

            Providing rays with different origins or random, incoherent directions
            with packet traversal enabled may lead to **incorrect results (false misses)**.
            If your rays do not meet these criteria, use `packet='Never'` to ensure
            correctness via scalar traversal.

        Args:
            origins (numpy.ndarray): A (N, 3) float array of ray origins.
            directions (numpy.ndarray): A (N, 3) float array of ray directions.
            t_max (numpy.ndarray, optional): A (N,) float array of maximum intersection distances.
            masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                             For a ray to test an instance for intersection, the bitwise
                                             AND of the ray's mask and the instance's mask must be non-zero.
                                             If not provided, rays default to mask 0xFFFF (intersect all instances).
            packet (PacketMode, optional): Choose packet usage strategy. Defaults to Auto.
            same_origin_eps (float, optional): Epsilon for same-origin test. Default 1e-6.
            max_spread (float, optional): Max spread allowed for a batch (cone angle, in degrees). Default 1.0.
            warn_on_incoherent (bool, optional): Warn when rays differ in origin. Default True.

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

    def is_occluded_batch(self,
                          origins: np.ndarray,
                          directions: np.ndarray,
                          t_max: Optional[np.ndarray] = None,
                          masks: Optional[np.ndarray] = None,
                          packet: PacketMode = PacketMode.Auto,
                          same_origin_eps: float = 1e-6,
                          max_spread: float = 1.0,
                          warn_on_incoherent: bool = True) -> np.ndarray:
        """
        Performs occlusion queries for a batch of rays, parallelized for performance.

        .. warning::
            Packet traversal (`Auto` or `Force`) is highly optimized but
            makes strict assumptions about the input rays:
                - Shared origin: All rays in a batch must share the same origin point.
                - Coherent directions: The ray directions should form a coherent frustum,
                    like rays cast from a camera through pixels on a screen.

            Providing rays with different origins or random, incoherent directions
            with packet traversal enabled may lead to **incorrect results (false misses)**.
            If your rays do not meet these criteria, use `packet='Never'` to ensure
            correctness via scalar traversal.

        Args:
            origins (numpy.ndarray): A (N, 3) float array of ray origins.
            directions (numpy.ndarray): A (N, 3) float array of ray directions.
            t_max (numpy.ndarray, optional): A (N,) float array of maximum occlusion distances.
                                             If a hit is found beyond this distance, it is ignored.
            masks (numpy.ndarray, optional): A (N,) uint32 array of per-ray visibility mask.
                                             For a ray to test an instance for intersection, the bitwise
                                             AND of the ray's mask and the instance's mask must be non-zero.
                                             If not provided, rays default to mask 0xFFFF (intersect all instances).
            packet (PacketMode, optional): Choose packet usage strategy. Defaults to Auto.
            same_origin_eps (float, optional): Epsilon for same-origin test. Default 1e-6.
            max_spread (float, optional): Max spread allowed for a batch (cone angle, in degrees). Default 1.0.
            warn_on_incoherent (bool, optional): Warn when rays differ in origin. Default True.

        Returns:
            numpy.ndarray: A boolean array of shape (N,) where `True` indicates occlusion.
        """
        ...

    def intersect_sphere(self, center: Vec3Like, radius: float) -> bool:
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

    def closest_point(self, point: Vec3Like) -> Optional[Dict[str, Any]]:
        """
        Finds the closest point on the BVH geometry to a given query point.

        This method performs a proximity search, useful for collision detection
        and geometry analysis. It traverses the BVH to find the nearest surface point.

        Args:
            point (Vec3Like): The 3D point to query from.

        Returns:
            Dict[str, Any] or None: A dictionary containing the query result:
                - 'point' (numpy.ndarray): The 3D coordinates of the closest point.
                - 'prim_id' (int): The ID of the primitive containing the closest point.
                - 'inst_id' (int): The ID of the instance (-1 for BLAS).
                - 'distance' (float): The Euclidean distance to the closest point.
            Returns None if the BVH is empty.
        """
        ...

    # Advanced manipulation methods

    def refit(self) -> None:
        """
        Refits the BVH to the current state of the source geometry, which is much
        faster than a full rebuild.

        Should be called after the underlying vertex data (numpy array used for construction)
        has been modified.

        Note: This will fail if the BVH was built with spatial splits (high-quality preset).
        """
        ...

    def optimize(self, iterations: int = 25, extreme: bool = False, stochastic: bool = False) -> None:
        """
        Optimizes the BVH tree structure to improve query performance.

        This is a costly operation best suited for static scenes. It works by
        re-inserting subtrees into better locations based on the SAH cost.

        Args:
            iterations (int): The number of optimization passes. Defaults to 25.
            extreme (bool): If true, a larger portion of the tree is considered
                            for optimization in each pass. Defaults to False.
            stochastic (bool): If true, uses a randomized approach to select
                               nodes for re-insertion. Defaults to False.
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

    def split_leaves(self, max_prims: int = 1) -> None:
        """
        Recursively splits leaf nodes until they contain at most `max_prims` primitives.
        This modifies the BVH in-place. Typically used to prepare a BVH for optimization
        by breaking it down into single-primitive leaves.

        Warning: This may fail if the BVH does not have enough pre-allocated node memory.

        Args:
            max_prims (int): The maximum number of primitives a leaf node can contain.
                             Defaults to 1.
        """
        ...

    def combine_leaves(self) -> None:
        """
        Merges adjacent leaf nodes if doing so improves the SAH cost.
        This modifies the BVH in-place. Typically used as a post-process after
        optimization to create more efficient leaves.

        Warning: This operation makes the internal primitive index array non-contiguous.

        It is highly recommended to call `compact()` after using this method to clean up
        the BVH structure.
        """
        ...

    def refit_tlas(self) -> None:
        """
        Refit / rebuild the TLAS using the current per-instance AABBs.

        Call this after one or more `set_instance_transform` / `set_instance_mask`
        updates. Faster than rebuilding BLASes. Does not change instance order.

        Raises ValueError if called on a BLAS (non-TLAS) BVH.
        """
        ...

    @property
    def instances(self) -> np.ndarray:
        """
        A writable 1D structured numpy view of the TLAS instances (zero-copy). Only meaningful on a TLAS.

        Notes:
            - Modify `transform` and/or `mask` in-place, then call `refit_tlas()`.
            - Changing `blas_id` is possible but advanced: it must reference an existing
              BLAS captured at TLAS build time. After changing it, call `update_instances`
              with `recompute_aabbs=True` (or call it once for all changed instances)
              before `refit_tlas()`, so AABBs are re-derived from the new BLAS.
        """
        ...

    def add_instance(self, transform: np.ndarray = ..., blas: BVH = ..., mask: int = 0xFFFFFFFF) -> None:
        """
        Adds a new instance to the TLAS. Only useable on a TLAS.

        This method stages the addition. You must call `refit_tlas()` afterwards to
        rebuild the acceleration structure and make the new instance visible to ray queries.

        Args:
            transform (numpy.ndarray): A (4, 4) float32, C-contiguous numpy array for the
                                       object-to-world transformation.
            blas (BVH): The Bottom-Level Acceleration Structure (a standard BVH) to instance.
            mask (int, optional): A 32-bit integer visibility mask. Defaults to 0xFFFFFFFF.
        """
        ...

    def remove_instance(self, instance_id: int) -> None:
        """
        Removes an instance from the TLAS by its index. Only useable on a TLAS.

        This method stages the removal. You must call `refit_tlas()` afterwards to
        rebuild the acceleration structure.

        Note:
            - Removing the last instance that refers to a particular BLAS does NOT
              de-register the BLAS. This is to avoid a costly re-indexing of all other
              instances. The BLAS simply becomes orphaned but remains available.
            - Raises `IndexError` if `instance_id` is out of range, `ValueError` if called on a BLAS.

        Args:
            instance_id (int): The index of the instance to remove.
        """
        ...

    def update_instances(self, instances: Optional[np.ndarray] = None, recompute_aabbs = True) -> None:
        """
        Bulk-updates TLAS instances from a structured array (zero extra copies in Python). Only useable on a TLAS.

        Args:
            instances (numpy.ndarray, optional): Structured array with dtype `instance_dtype`
                Must contain fields:
                  - 'transform' : (4, 4) float32, row-major
                  - 'blas_id'   : uint32
                  - 'mask'      : uint32 (Optional. Defaults preserved if omitted)
            recompute_aabbs (bool): If True (default), recomputes each instance's inverse transform and world-space AABB.

        Notes:
            - After updates, call `refit_tlas()` to rebuild the TLAS over the updated AABBs.
            - The array length must match the existing TLAS instance count.
        """
        ...

    def compact_blases(self) -> None:
        """
        Removes any unused (orphaned) BLASes from the TLAS's internal lists.

        This is a potentially slow operation as it requires re-indexing all existing
        instances. It should only be called when memory usage is a critical concern
        after many instances have been removed.

        After calling this, you must still call `refit_tlas()` to rebuild the
        acceleration structure.
        """
        ...

    def set_instance_transform(self, i: int, m4x4: np.ndarray) -> None:
        """
        Update the transform of one TLAS instance and recompute its world-space AABB.

        Args:
            i (int): Index of the instance to update (0-based).
            m4x4 (numpy.ndarray): Row-major object-to-world transform.
                                  he inverse is computed internally.

        Notes: - This does not rebuild the TLAS. You must call `refit_tlas()` after a batch of updates.
               - Raises `IndexError` if `i` is out of range, `ValueError` if called on a BLAS.
        """
        ...

    def set_traversal_cost(self, cost: float) -> None:
        """
        Sets the SAH traversal cost used by metrics/optimizers.

        This does not rebuild the BVH. It simply updates the cost used by SAH reporting
        and any subsequent optimization passes that consult it.
        """
        ...

    def set_intersection_cost(self, cost: float) -> None:
        """
        Sets the SAH intersection cost used by metrics/optimizers.

        This does not rebuild the BVH. It simply updates the cost used by SAH reporting
        and any subsequent optimization passes that consult it.
        """
        ...

    def set_instance_mask(self, i: int, mask: int) -> None:
        """
        Set the 32-bit visibility mask for an instance.

        Args:
            i (int): Instance index (0-based).
            mask (int): Bitmask to AND with ray masks during traversal.
        """
        ...

    def set_opacity_maps(self, map_data: np.ndarray, N: int) -> None:
        """
        Sets the opacity micro-maps for alpha testing during intersection.

        The BVH must be built before calling this. The intersection queries will
        automatically use the map to discard hits on transparent parts of triangles.

        Args:
            map_data (numpy.ndarray): A 1D uint32 numpy array containing the packed
                                      bitmasks for all triangles. The size must be
                                      (tri_count * N * N + 31) // 32
            N (int): The resolution of the micro-map per triangle (e.g., 8 for an 8x8 grid).
        """
        ...

    def clear_opacity_maps(self) -> None:
        """
        Detach opacity micromaps from this BVH.

        After calling this, ray tests ignore per-triangle alpha coverage.
        Safe to call when no micromaps are present (no-op).
        """
        ...

    @property
    def has_opacity_maps(self) -> bool:
        """Returns True is opacity micromaps are attached to this BVH."""
        ...

    @property
    def opacity_map_level(self) -> int:
        """Opacity micromap subdivision level `N` (0 if none)."""
        ...

    @property
    def nodes(self) -> np.ndarray:
        """The structured numpy array of BVH nodes (only for standard layout)."""
        ...

    @property
    def prim_indices(self) -> np.ndarray:
        """The BVH-ordered array of primitive indices (only for standard layout)."""
        ...

    @property
    def source_vertices(self) -> Optional[np.ndarray]:
        """The source vertex buffer as a numpy array, or None."""
        ...

    @property
    def source_indices(self) -> Optional[np.ndarray]:
        """The source index buffer for an indexed mesh as a numpy array, or None."""
        ...

    @property
    def source_aabbs(self) -> Optional[np.ndarray]:
        """The source AABB buffer as a numpy array, or None."""
        ...

    @property
    def source_points(self) -> Optional[np.ndarray]:
        """The source point buffer for sphere geometry as a numpy array, or None."""
        ...

    @property
    def sphere_radius(self) -> Optional[float]:
        """The radius for sphere geometry, or None."""
        ...

    def get_buffers(self) -> Dict[str, np.ndarray]:
        """
        Returns a dict of zero-copy numpy views over the active BVH and its source geometry.

            Common keys:
                - nodes          : The node buffer for the active layout. 2D float32 for structured
                                   layouts (Standard: (N, 8), BVH_GPU/SoA: (N,16), MBVH4/8: (N,K)).
                                   For blocked layouts where nodes are exported as flat blocks, 'nodes'
                                   is a 1D float32 alias to 'packed_data'.
                - prim_indices   : Reordered leaf mapping stored by tinybvh (uint32). For BLAS this maps
                                   leaf → primitive id; for TLAS this maps leaf → instance id.
                - leaf_ids       : Alias of 'prim_indices' (stable name to target in shaders).
                - primitives     : Geometry-agnostic alias to the primitive stream:
                                      - Triangles : 'indices' if present, otherwise 'vertices'
                                      - AABBs     : 'aabbs'
                                      - Spheres   : 'points'
                - primitive_kind : One of {"Triangles", "AABBs", "Spheres"}.

            Geometry keys:
                Triangles:
                - vertices      : (V, 3) float32, Vertex positions (zero-copy reference to source array)
                - indices       : (T, 3) uint32, Triangle indices (optional for non-indexed meshes)

                AABBs:
                - aabbs         : (N, 2, 3) float32, Boxes extents (min, max)
                - inv_extents   : (N, 3) float32 (optional), Inverse of boxes extents

                Spheres:
                - points        : (N, 3) float32, Sphere centers
                - sphere_radius : float scalar, Radius (broadcast)

                TLAS:
                - instances     : Structured array of instances (zero-copy view), only present for TLAS.

            Layout-specific keys:
                - packed_data : 1D float32, View of blocked node memory for BVH4/8 CPU/GPU layouts.
                                BVH4/8 CPU: 64B blocks → 16 floats per block
                                BVH4_GPU/CWBVH: 16B blocks → 4 floats per block
                - triangles   : For CWBVH only. The embedded triangle stream used by the layout.
                                12 or 16 floats per triangle, depending on build flags

            Notes:
                - All arrays are views into internal memory, no copies are made.
                - The exact shapes of 'nodes' depend on the layout exporter (see above).
                - For blocked layouts, 'nodes' is an alias to 'packed_data'.

        Returns:
            Dict[str, numpy.ndarray]: A dictionary mapping buffer names to their corresponding raw data arrays.
        """
        ...

    def get_SSBO_bundle(self) -> Dict[str, np.ndarray]:
        """
        Returns a consistent (layout-agnostic) dict for SSBO uploads and GLSL setup:

        Keys:
            - node_buffer        : Raw bytes view of the active layout's node memory (for SSBO)
            - node_key           : Source key used for nodes ("nodes" or "packed_data")
            - node_count         : Number of structured nodes (2D layouts), else 0
            - block_count        : Number of node blocks (blocked layouts), else 0
            - node_stride_bytes  : Bytes per structured node row (when nodes are 2D)
            - block_stride_bytes : Bytes per node block for blocked layouts (e.g., 16 or 64) when applicable
            - leaf_ids           : Reordered leaf mapping (leaf → prim id for BLAS, leaf → instance id for TLAS)
            - primitives         : Geometry-agnostic primitive stream:
                                    Triangles → 'indices' if present, else 'vertices'
                                    AABBs → 'aabbs'
                                    Spheres → 'points'
            - index_buffer       : Triangle index buffer (if the source mesh is indexed)
            - vertices           : Triangle vertex positions
            - aabbs              : AABB boxes (min,max)
            - points             : Sphere centers
            - sphere_radius      : Sphere radius (scalar)
            - primitive_kind     : "Triangles", "AABBs" or "Spheres"
            - embedded_triangles : CWBVH-only triangle stream (if present)
            - instances          : TLAS instance array (only for TLAS)
            - layout             : String version of the Layout
            - geometry_type      : String version of the geometry type
            - is_tlas            : Whether the BVH is a TLAS or not
            - defines            : Dict of "TBVH_*" macros for GLSL
            - preamble           : String of "#define TBVH_* ..." lines to prepend to shaders

        Returns:
            Dict[str, numpy.ndarray]: A dictionary containing everything needed for SSBO upload
        """
        ...

    @property
    def node_count(self) -> int:
        """Total number of nodes in the currently active BVH representation."""
        ...

    @property
    def leaf_count(self) -> int:
        """Total number of leaf nodes (only for standard layout)."""
        ...

    @property
    def prim_count(self) -> int:
        """Total number of primitives in the BVH."""
        ...

    @property
    def instance_count(self) -> int:
        """Total number of instances in this TLAS. Only meaningful if TLAS."""
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
    def is_tlas(self) -> bool:
        """Returns True if the BVH is a Top-Level Acceleration Structure (TLAS)."""
        ...

    @property
    def is_blas(self) -> bool:
        """Returns True if the BVH is a Bottom-Level Acceleration Structure (BLAS)."""
        ...

    @property
    def geometry_type(self) -> GeometryType:
        """The type of underlying geometry the BVH was built on."""
        ...

    @property
    def is_compact(self) -> bool:
        """Returns True if the BVH is contiguous in memory."""
        ...

    @property
    def is_refittable(self) -> bool:
        """Returns True if the BVH can be refitted."""
        ...

    @property
    def traversal_cost(self) -> float:
        """The traversal cost used in the Surface Area Heuristic (SAH) calculation during the build."""
        ...

    @property
    def intersection_cost(self) -> float:
        """The intersection cost used in the Surface Area Heuristic (SAH) calculation during the build."""
        ...

    @property
    def sah_cost(self) -> float:
        """Calculates the Surface Area Heuristic (SAH) cost of the BVH."""
        ...

    @property
    def epo_cost(self) -> float:
        """Calculates the Expected Projected Overlap (EPO) cost of the BVH (only for standard layout)."""
        ...

    @property
    def layout(self) -> Layout:
        """The current active memory layout of the BVH."""
        ...

    @property
    def cached_layouts(self) -> List[Layout]:
        """A list of the BVH layouts currently held in the cache."""
        ...

    @property
    def memory_usage(self) -> Dict[str, Any]:
        """
        Reports a detailed breakdown of the memory usage for the BVH.

        The returned dictionary contains the following keys:

        - `total_bytes` (int):
            The total memory footprint, including C++ buffers and referenced Python objects.
        - `base_bvh` (Dict):
            Memory used by the mandatory standard layout BVH. Contains `{'bytes': int}`.
        - `cached_layouts` (Dict, optional):
            A dictionary mapping layout names to their memory usage in bytes.
        - `tlas_data` (Dict, optional):
            Memory used by internal C++ TLAS buffers (instances, pointers). Contains `{'bytes': int}`.
        - `source_geometry_bytes` (List[int], optional):
            A list of byte sizes for each external Python object this BVH holds a
            reference to. This includes source NumPy arrays (vertices, indices) and,
            for a TLAS, the total memory footprint of each referenced BLAS

        Returns:
            Dict[str, Any]: A dictionary detailing the memory usage in bytes.
        """
        ...

class BVHVerbose:
    """
    Editable/inspectable BVH representation.

    This layout exposes explicit nodes and indices for diagnostics, SAH analysis,
    local optimizations, and maintenance (refit / compaction).

    Convert from/to the compact `BVH` with `from_bvh()` and `to_bvh()`.
    """

    @staticmethod
    def from_bvh(bvh: BVH, compact: bool = True) -> BVHVerbose:
        """
        Creates a verbose BVH from a compact `BVH`.

        Args:
            bvh (BVH): Source BVH to convert.
            compact (bool): If True (default), compacts nodes during conversion.

        Returns:
            BVHVerbose: New verbose tree (deep copy).
        """
        ...

    @staticmethod
    def to_bvh(compact: bool = True) -> BVH:
        """
        Converts this verbose BVH back to a compact `BVH`.

        Args:
            compact (bool): Compacts the resulting BVH layout.

        Returns:
            A new compact BVH (deep copy).
        """
        ...

    def sah_cost_at(self, node: int = 0) -> int:
        """
        Computes the Surface Area Heuristic (SAH) cost for a node or the whole tree.

        Args:
            node (int): Node index to evaluate (0 = root).

        Returns:
            int
        """
        ...

    @property
    def node_count(self) -> int:
        """Total number of nodes in the verbose tree."""
        ...

    # @property
    # def prim_count(self) -> int:
    #     """Total number of nodes in the verbose tree."""
    #     ...

    @property
    def sah_cost(self) -> int:
        """Computes the Surface Area Heuristic (SAH) cost of the verbose tree."""
        ...

    # def prim_count_at(self, node: int = 0) -> int:
    #     """
    #     Number of primitives referenced under a node.
    #
    #     Args:
    #         node (int): Node index (0 = root).
    #
    #     Returns:
    #         int
    #     """
    #     ...

    def refit(self, node: int = 0, skip_leaves: bool = False) -> None:
        """
        Refits the AABBs starting at `node`.

        Args:
            node (int): Subtree root to refit (0 = entire tree).
            skip_leaves (bool): If True, leaf bounds are assumed up to date and are not recomputed (default False).
        """
        ...

    def compact(self) -> None:
        """
        Removes dead/degenerate nodes and pack arrays tightly.

        Useful after structural edits to ensure a compact representation.
        """
        ...

    def sort_indices(self) -> None:
        """
        Reorders primitive indices to improve spatial/streaming coherence.

        This can improve downstream cache behaviour.
        """
        ...

    def optimize(self, iterations: int = 0, extreme: bool = False, stochastic: bool = False) -> None:
        """
        Runs local BVH optimizations to reduce SAH cost.

        Args:
            iterations (int): Number of improvement passes (default 25).
            extreme (bool): Enables more aggressive (but slower) transformations. Default False.
            stochastic (bool): Adds randomness to escape local minima. Default False.
        """
        ...