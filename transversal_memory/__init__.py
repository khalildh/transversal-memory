"""
Transversal Memory
==================
Content-addressable memory via projective geometry.

Quick start:

    from transversal_memory import P3Memory, GramMemory, ProjectedMemory
    from transversal_memory.plucker import random_line, line_from_points
    from transversal_memory.embeddings import WordMemory, random_embeddings
"""

from .plucker import (
    line_from_points,
    line_from_direction_moment,
    line_from_dm_vec,
    project_to_line,
    project_to_line_dual,
    plucker_inner,
    plucker_relation,
    is_valid_line,
    lines_meet,
    random_line,
    random_projection,
    random_projection_dual,
    find_transversals,
)

from .solver import solve_p3, solve_general

from .memory import P3Memory, GramMemory, ProjectedMemory

__all__ = [
    "line_from_points",
    "line_from_direction_moment",
    "line_from_dm_vec",
    "project_to_line",
    "project_to_line_dual",
    "plucker_inner",
    "plucker_relation",
    "is_valid_line",
    "lines_meet",
    "random_line",
    "random_projection",
    "random_projection_dual",
    "find_transversals",
    "solve_p3",
    "solve_general",
    "P3Memory",
    "GramMemory",
    "ProjectedMemory",
]

from .cooccurrence import (
    CooccurrenceMatrix,
    SVDEmbeddings,
    embeddings_from_associations,
)
