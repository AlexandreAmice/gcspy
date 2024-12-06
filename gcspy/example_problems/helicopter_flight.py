import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexSets


def make_helicopter_flight_problem(
    helicopter_initial_position,
    helicopter_desired_position,
    num_islands,
    islands_max_radius,
    helicopter_speed,
    battery_decrease_rate,
    battery_recharge_rate,
    max_battery_charge,
    numpy_seed=None,
    ax=None,
):
    """
    Plan the flight of a helicopter from helicopter_initial_position to
    helicopter_desired_position. The helicopter flies at a constant speed
    helicopter_speed with a battery that decreases at a rate
    battery_decrease_rate with a maximum capacity of max_battery_charge. To
    recharge the battery the helicopter must land on one of the num_island
    islands to recharge its battery at a rate of battery_recharge_rate. The
    objective is to arrive at the target as fast as possible.

    Solve this problem as a shortest path problem.
    """
    C = []
    r = []
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    while len(C) < num_islands:
        ci = np.random.rand(2)
        ri = np.random.rand() * islands_max_radius
        keep = True
        for cj, rj in zip(C, r):
            dij = np.linalg.norm(ci - cj)
            if dij < ri + rj:
                keep = False
                break
        if keep:
            C.append(ci)
            r.append(ri)
    C = np.array(C)
    r = np.array(r)

    gcs = GraphOfConvexSets()

    s = gcs.add_vertex("s")
    qs = s.add_variable(2)
    s.add_constraint(qs == 0)
    zs = s.add_variable(1)
    s.add_constraint(zs == max_battery_charge)

    t = gcs.add_vertex("t")
    qt = t.add_variable(2)
    t.add_constraint(qt == 1)
    zt = t.add_variable(1)
    t.add_constraint(zt >= 0)
    t.add_constraint(zt <= max_battery_charge)

    for i in range(num_islands):
        vi = gcs.add_vertex(f"v{i}")
        # helicopter positions
        qi = vi.add_variable(2)
        vi.add_constraint(cp.norm(qi - C[i], 2) <= r[i])

        # helicopter battery charge on arrival
        zi0 = vi.add_variable(1)
        vi.add_constraint(zi0 >= 0)
        vi.add_constraint(zi0 <= max_battery_charge)

        # helicopter battery charge on departure
        zi1 = vi.add_variable(1)
        vi.add_constraint(zi1 >= 0)
        vi.add_constraint(zi1 <= max_battery_charge)

        # time spent on island
        ti = vi.add_variable(1)
        vi.add_constraint(ti >= 0)
        vi.add_constraint(ti <= max_battery_charge / battery_recharge_rate)
        vi.add_cost(ti)
        vi.add_constraint(zi1 == zi0 + ti * battery_recharge_rate)

    max_flight_distance = max_battery_charge / battery_decrease_rate * helicopter_speed
    for i in range(num_islands):
        vi = gcs.get_vertex_by_name(f"v{i}")
        qi, zi0, zi1, ti = vi.variables

        ds = np.linalg.norm(C[i] - helicopter_initial_position)
        if ds < max_flight_distance + r[i]:
            qs, zs = s.variables
            e = gcs.add_edge(s, vi)
            tsi = cp.norm(qi - qs, 2) / helicopter_speed
            e.add_cost(tsi)
            e.add_constraint(zi0 <= zs - battery_decrease_rate * tsi)

        dt = np.linalg.norm(C[i] - helicopter_desired_position)
        if dt < max_flight_distance + r[i]:
            qt, zt = t.variables
            e = gcs.add_edge(vi, t)
            tti = cp.norm(qi - qt, 2) / helicopter_speed
            e.add_cost(tti)
            e.add_constraint(zt <= zi1 - battery_decrease_rate * tti)

        for j in range(num_islands):
            if i != j:
                dij = np.linalg.norm(C[i] - C[j])
                if dij < max_flight_distance + r[i] + r[j]:
                    vj = gcs.get_vertex_by_name(f"v{j}")
                    qj, zj0, zj1, tj = vj.variables
                    e = gcs.add_edge(vi, vj)
                    tij = cp.norm(qi - qj, 2) / helicopter_speed
                    e.add_cost(tij)
                    e.add_constraint(zj0 <= zi1 - battery_decrease_rate * tij)

    if ax is not None:
        ax.gca().add_patch(
            ax.Rectangle(
                (-islands_max_radius, -islands_max_radius),
                1 + 2 * islands_max_radius,
                1 + 2 * islands_max_radius,
                fc="azure",
            )
        )
        for i in range(num_islands):
            ax.gca().add_patch(ax.Circle(C[i], r[i], ec="k", fc="lightgreen"))
        ax.gca().set_aspect("equal")
        ax.xticks([])
        ax.yticks([])
        ax.xlim([-islands_max_radius, 1 + islands_max_radius])
        ax.ylim([-islands_max_radius, 1 + islands_max_radius])
    if ax is None:
        return gcs
    return gcs, ax
