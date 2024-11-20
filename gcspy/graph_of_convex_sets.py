import cvxpy as cp
import numpy as np
from collections.abc import Iterable
from gcspy.programs import ConvexProgram, ConicProgram
from gcspy.graph_problems import (
    graph_problem,
    ilp_translator,
    shortest_path,
    traveling_salesman,
    spanning_tree,
    facility_location,
)


class Vertex(ConvexProgram):

    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def _verify_variables(self, variables):
        ids0 = {variable.id for variable in self.variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this vertex.")

    def get_feasible_point(self, *args, **kwargs):
        """
        :param args: arguments forwarded to cvxpy
        :param kwargs: kwargs forwarded to cvxpy
        :return:
        """
        values = [variable.value for variable in self.variables]
        prob = cp.Problem(cp.Minimize(0), self.constraints)
        prob.solve(*args, **kwargs)
        feasible_point = [variable.value for variable in self.variables]
        for variable, value in zip(self.variables, values):
            variable.value = value
        return feasible_point


class Edge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.conic_program = None
        self.name = (self.tail.name, self.head.name)

    def _verify_variables(self, variables):
        edge_variables = self.variables + self.tail.variables + self.head.variables
        ids0 = {variable.id for variable in edge_variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this edge.")


class GraphOfConvexSets:

    def __init__(self):
        self.vertices = []
        self.edges = []

    def add_vertex(self, name=""):
        vertex = Vertex(name)
        self.vertices.append(vertex)
        return vertex

    def add_edge(self, tail, head):
        edge = Edge(tail, head)
        self.edges.append(edge)
        return edge

    def add_subgraph(self, gcs):
        self.vertices += gcs.vertices
        self.edges += gcs.edges

    def get_edge(self, tail, head):
        for edge in self.edges:
            if edge.tail == tail and edge.head == head:
                return edge

    def get_vertex_by_name(self, name):
        for vertex in self.vertices:
            if vertex.name == name:
                return vertex
        raise ValueError(f"There is no vertex named {name}.")

    def get_edge_by_name(self, tail_name, head_name):
        for edge in self.edges:
            if edge.tail.name == tail_name and edge.head.name == head_name:
                return edge
        raise ValueError(
            f"There is no edge with tail named {tail_name} and head named {head_name}."
        )

    def vertex_index(self, vertex):
        return self.vertices.index(vertex)

    def edge_index(self, edge):
        return self.edges.index(edge)

    def incoming_edges(self, v):
        if isinstance(v, Vertex):
            return [e for e in self.edges if e.head == v]
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.head in v and e.tail not in v]

    def outgoing_edges(self, v):
        if isinstance(v, Vertex):
            return [e for e in self.edges if e.tail == v]
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.tail in v and e.head not in v]

    def incident_edges(self, v):
        return self.incoming_edges(v) + self.outgoing_edges(v)

    def incoming_indices(self, v):
        if isinstance(v, Vertex):
            return [k for k, e in enumerate(self.edges) if e.head == v]
        if isinstance(v, Iterable):
            return [
                k for k, e in enumerate(self.edges) if e.head in v and e.tail not in v
            ]

    def outgoing_indices(self, v):
        if isinstance(v, Vertex):
            return [k for k, e in enumerate(self.edges) if e.tail == v]
        if isinstance(v, Iterable):
            return [
                k for k, e in enumerate(self.edges) if e.tail in v and e.head not in v
            ]

    def incident_indices(self, v):
        return self.incoming_indices(v) + self.outgoing_indices(v)

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def vertex_binaries(self):
        return np.array([vertex.y for vertex in self.vertices])

    def edge_binaries(self):
        return np.array([edge.y for edge in self.edges])

    def to_conic(self):
        for vertex in self.vertices:
            vertex.to_conic()
        for edge in self.edges:
            edge.to_conic()

    def solve_shortest_path(self, source, target, *args, **kwargs):
        problem = lambda *local_args: shortest_path(*local_args, s=source, t=target)
        return graph_problem(self, problem, *args, **kwargs)

    def solve_traveling_salesman(self, subtour_elimination=True, *args, **kwargs):
        problem = lambda *args: traveling_salesman(
            *args, subtour_elimination=subtour_elimination
        )
        return graph_problem(self, problem, *args, **kwargs)

    def solve_spanning_tree(self, root, subtour_elimination=True, *args, **kwargs):
        problem = lambda *args: spanning_tree(
            *args, root=root, subtour_elimination=subtour_elimination
        )
        return graph_problem(self, problem, *args, **kwargs)

    def solve_facility_location(self, *args, **kwargs):
        return graph_problem(self, facility_location, *args, **kwargs)

    def solve_from_ilp(self, ilp_constraints, callback=None, *args, **kwargs):
        problem = lambda *args: ilp_translator(*args, ilp_constraints=ilp_constraints)
        return graph_problem(self, problem, callback=callback, *args, **kwargs)

    def get_convex_restriction(self, vertex_indices, edge_indices, *args, **kwargs):
        for k in edge_indices:
            edge = self.edges[k]
            i = self.vertex_index(edge.tail)
            j = self.vertex_index(edge.head)
            if i not in vertex_indices or j not in vertex_indices:
                raise ValueError("Given indices do not form a subgraph.")
        cost = 0
        constraints = []
        for i in vertex_indices:
            vertex = self.vertices[i]
            cost += vertex.cost
            constraints.extend(vertex.constraints)
        for k in edge_indices:
            edge = self.edges[k]
            cost += edge.cost
            constraints.extend(edge.constraints)
        return cp.Problem(cp.Minimize(cost), constraints)

    def solve_convex_restriction(self, vertex_indices, edge_indices, *args, **kwargs):
        prob = self.get_convex_restriction(
            vertex_indices, edge_indices, *args, **kwargs
        )
        prob.solve(*args, **kwargs)
        for i, vertex in enumerate(self.vertices):
            vertex.y.value = 1 if i in vertex_indices else None
        for k, edge in enumerate(self.edges):
            edge.y.value = 1 if k in edge_indices else None
        return prob

    def graphviz(self):
        from gcspy.plot_utils import graphviz_gcs

        return graphviz_gcs(self)

    def plot_2d(self, **kwargs):
        from gcspy.plot_utils import plot_gcs_2d

        return plot_gcs_2d(self, **kwargs)

    def plot_subgraph_2d(self):
        from gcspy.plot_utils import plot_subgraph_2d

        return plot_subgraph_2d(self)
