"""
Mesh utilities for SAM3D output.
"""

from typing import Dict, List, Sequence, Union
import math


def create_placeholder_mesh() -> Dict[str, List[float]]:
    """Return a low-poly human-like mesh placeholder.

    This is a stand-in for SAM 3D Body output until the real model is wired.
    """
    # Simple body + head box mesh (vertices in XYZ order)
    body = [
        (-0.4, -1.0, -0.2),
        (0.4, -1.0, -0.2),
        (0.4, 0.6, -0.2),
        (-0.4, 0.6, -0.2),
        (-0.4, -1.0, 0.2),
        (0.4, -1.0, 0.2),
        (0.4, 0.6, 0.2),
        (-0.4, 0.6, 0.2),
    ]

    head = [
        (-0.2, 0.6, -0.15),
        (0.2, 0.6, -0.15),
        (0.2, 1.0, -0.15),
        (-0.2, 1.0, -0.15),
        (-0.2, 0.6, 0.15),
        (0.2, 0.6, 0.15),
        (0.2, 1.0, 0.15),
        (-0.2, 1.0, 0.15),
    ]

    vertices = body + head

    def box_faces(offset: int) -> List[int]:
        return [
            offset + 0, offset + 1, offset + 2,
            offset + 2, offset + 3, offset + 0,
            offset + 4, offset + 5, offset + 6,
            offset + 6, offset + 7, offset + 4,
            offset + 0, offset + 4, offset + 7,
            offset + 7, offset + 3, offset + 0,
            offset + 1, offset + 5, offset + 6,
            offset + 6, offset + 2, offset + 1,
            offset + 3, offset + 2, offset + 6,
            offset + 6, offset + 7, offset + 3,
            offset + 0, offset + 1, offset + 5,
            offset + 5, offset + 4, offset + 0,
        ]

    faces = box_faces(0) + box_faces(len(body))

    flat_vertices = [coord for vertex in vertices for coord in vertex]

    return {
        "format": "sam3d-body",
        "vertices": flat_vertices,
        "faces": faces,
    }


def _flatten_vertices(vertices: Union[Sequence[float], Sequence[Sequence[float]]]) -> List[float]:
    if vertices is None:
        return []

    if hasattr(vertices, "shape"):
        return [float(x) for x in vertices.reshape(-1)]

    if len(vertices) == 0:
        return []

    first = vertices[0]
    if isinstance(first, (list, tuple)) or hasattr(first, "shape"):
        return [float(coord) for vertex in vertices for coord in vertex]

    return [float(x) for x in vertices]


def _flatten_faces(faces: Union[Sequence[int], Sequence[Sequence[int]]]) -> List[int]:
    if faces is None:
        return []

    if hasattr(faces, "shape"):
        return [int(x) for x in faces.reshape(-1)]

    if len(faces) == 0:
        return []

    first = faces[0]
    if isinstance(first, (list, tuple)) or hasattr(first, "shape"):
        return [int(idx) for face in faces for idx in face]

    return [int(x) for x in faces]


def serialize_mesh(
    vertices: Union[Sequence[float], Sequence[Sequence[float]]],
    faces: Union[Sequence[int], Sequence[Sequence[int]]],
    mesh_format: str = "sam3d-body",
) -> Dict[str, List[float]]:
    return {
        "format": mesh_format,
        "vertices": _flatten_vertices(vertices),
        "faces": _flatten_faces(faces),
    }


def compute_bounds(vertices: Union[Sequence[float], Sequence[Sequence[float]]]) -> Dict[str, List[float]]:
    flat_vertices = _flatten_vertices(vertices)
    if not flat_vertices:
        return {"center": [0.0, 0.0, 0.0], "radius": 1.0}

    xs = flat_vertices[0::3]
    ys = flat_vertices[1::3]
    zs = flat_vertices[2::3]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center = [
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        (min_z + max_z) / 2.0,
    ]

    radius = 0.0
    for i in range(0, len(flat_vertices), 3):
        dx = flat_vertices[i] - center[0]
        dy = flat_vertices[i + 1] - center[1]
        dz = flat_vertices[i + 2] - center[2]
        radius = max(radius, math.sqrt(dx * dx + dy * dy + dz * dz))

    return {"center": center, "radius": radius}
