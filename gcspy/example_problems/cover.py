import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexSets


def make_set_cover_problem(mesh, n_spheres, r_max, alpha):
    """
    Given a mesh of 2D triangles, cover the vertices of the mesh with at most
    n_spheres of radius no larger than r_max such that each triangle is completely covered by at least one
    sphere.

    One pays a fixed cost of alpha for every sphere used as well as a cost equal to the area of that sphere.
    The objective is to minimize this cost.

    Solve this problem as a facility location problem.
    """
    x_min = np.full(2, np.inf)
    x_max = -x_min
    for triangle in mesh:
        x_min = np.min([x_min, np.min(triangle, axis=0)], axis=0)
        x_max = np.max([x_max, np.max(triangle, axis=0)], axis=0)

    gcs = GraphOfConvexSets()

    # fixed cost of adding a sphere

    spheres = []
    for i in range(n_spheres):
        s = gcs.add_vertex(f"s{i}")
        c = s.add_variable(2)
        r = s.add_variable(1)
        s.add_constraint(c >= x_min)
        s.add_constraint(c <= x_max)
        s.add_constraint(r >= 0)
        s.add_constraint(r <= r_max)
        s.add_cost(alpha)
        s.add_cost(np.pi * r**2)
        spheres.append(s)

    triangles = []
    for i in range(len(mesh)):
        t = gcs.add_vertex(f"t{i}")
        t.add_constraint(t.add_variable(1)[0] == 0)
        triangles.append(t)

    for s in spheres:
        c = s.variables[0]
        r = s.variables[1]
        for i, t in enumerate(triangles):
            edge = gcs.add_edge(s, t)
            for p in mesh[i]:
                edge.add_constraint(cp.norm(p - c, 2) <= r)
    return gcs
