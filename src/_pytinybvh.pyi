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
    AABBs: ClassVar[GeometryType]
    Spheres: ClassVar[GeometryType]

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
    """Use packets only if rays have a shared origin. Assumes coherent directions."""

    Never: ClassVar[PacketMode]
    """Always use scalar traversal. Safest for non-coherent rays."""

    Force: ClassVar[PacketMode]
    """
    Force packet traversal. This can provide a speedup, but is unsafe
    if rays are not coherent (even if they have the same origin).
    """
    def __int__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


# Main classes

class Ray:
    """Represents a ray for intersection queries."""
    origin: Vec3Like
    direction: Vec3Like
    t: float
    mask: int

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

        This method uses multi-core parallelization for all geometry types (triangles, AABBs, spheres).

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
        Returns a dictionary of raw, zero-copy numpy arrays for all current BVH's internal data and geometry.

        This provides low-level access to all the underlying C++ buffers. The returned arrays
        are views into the BVH's memory.
        The structure of the returned dictionary and the shape/content of the arrays depend on the active layout.

        This is primarily useful for advanced use cases (sending BVH data to a SSBO, etc.).

        Returns:
            Dict[str, numpy.ndarray]: A dictionary mapping buffer names (e.g., 'nodes',
                                    'prim_indices', 'packed_data', 'vertices', etc.) to
                                    their corresponding raw data arrays.
        """
        ...

    @property
    def node_count(self) -> int:
        """Total number of nodes in the currently active BVH representation."""
        ...

    @property
    def leaf_count(self) -> int:
        """The total number of leaf nodes (only for standard layout)."""
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