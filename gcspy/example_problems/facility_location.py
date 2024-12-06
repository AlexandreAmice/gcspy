import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexSets


def facility_location_gcs(facilities, users):
    gcs = GraphOfConvexSets()

    for facility in facilities:
        for user in users:
            edge = gcs.add_edge(facility, user)
            edge.add_cost(cp.norm(facility.variables[0] - user.variables[0], 2))
