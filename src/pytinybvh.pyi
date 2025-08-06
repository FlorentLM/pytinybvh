from __future__ import annotations
import numpy as np


bvh_node_dtype: np.dtype


class vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float = ..., y: float = ..., z: float = ...) -> None: ...


class Ray:
    origin: vec3
    direction: vec3
    t: float

    @property
    def u(self) -> float: ...

    @property
    def v(self) -> float: ...

    @property
    def prim_id(self) -> int: ...

    def __init__(self, origin: vec3, direction: vec3, t: float = ...) -> None: ...


class BVH:
    @staticmethod
    def from_triangles(triangles: np.ndarray) -> BVH: ...

    @staticmethod
    def from_points(points: np.ndarray, radius: float = ...) -> BVH: ...

    @staticmethod
    def from_vertices_4f(vertices: np.ndarray) -> BVH: ...

    def intersect(self, ray: Ray) -> bool: ...

    def refit(self) -> None: ...

    @property
    def nodes(self) -> np.ndarray: ...

    @property
    def prim_indices(self) -> np.ndarray: ...

    @property
    def node_count(self) -> int: ...

    @property
    def prim_count(self) -> int: ...

    @property
    def aabb_min(self) -> vec3: ...

    @property
    def aabb_max(self) -> vec3: ...