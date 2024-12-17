from gcspy.programs import ConicProgram


def shortest_path(gcs, xv, zv, ze_out, ze_inc, s, t):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_indices(vertex)
        out_edges = gcs.outgoing_indices(vertex)

        if vertex == s:
            constraints.append(sum(ye[inc_edges]) == 0)
            constraints.append(sum(ye[out_edges]) == 1)
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == xv[i])

        elif vertex == t:
            constraints.append(sum(ye[out_edges]) == 0)
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints.append(zv[i] == xv[i])

        else:
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(yv[i] <= 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints += vertex.conic.eval_constraints(xv[i] - zv[i], 1 - yv[i])

    return constraints


def get_shortest_path_constraints(gcs, s, t):
    """
    Returns a dictionary mapping gcs vertices to vertex-separable constraints. The
    special value None is all the constraints which are not vertex separable
    """
    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    vertex_separable_constraints = dict()
    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        vertex_separable_constraints[vertex] = []
        inc_edges = gcs.incoming_indices(vertex)
        out_edges = gcs.outgoing_indices(vertex)

        if vertex == s:
            vertex_separable_constraints[vertex].append(sum(ye[inc_edges]) == 0)
            vertex_separable_constraints[vertex].append(sum(ye[out_edges]) == 1)
            vertex_separable_constraints[vertex].append(yv[i] == sum(ye[out_edges]))

        elif vertex == t:
            vertex_separable_constraints[vertex].append(sum(ye[out_edges]) == 0)
            vertex_separable_constraints[vertex].append(sum(ye[inc_edges]) == 1)
            vertex_separable_constraints[vertex].append(yv[i] == sum(ye[inc_edges]))

        else:
            vertex_separable_constraints[vertex].append(yv[i] == sum(ye[out_edges]))
            vertex_separable_constraints[vertex].append(yv[i] == sum(ye[inc_edges]))
        vertex_separable_constraints[vertex].append(yv[i] <= 1)
        vertex_separable_constraints[vertex].append(yv[i] >= 0)
        constraints += vertex_separable_constraints[vertex]

    return constraints, vertex_separable_constraints
