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
import networkx as nx
import typing


class Vertex(ConvexProgram):

    def __init__(self, name="", convex_relaxation=False):
        super().__init__(name, convex_relaxation)
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

    def __repr__(self):
        return f"{self.name}: ({self.variables})"

    def __str__(self):
        return self.__repr__()


class Edge(ConvexProgram):

    def __init__(self, tail, head, convex_relaxation=False):
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)
        super().__init__(self.name, convex_relaxation)

    @property
    def edge_variables(self):
        return self.variables + self.tail.variables + self.head.variables

    def _verify_variables(self, variables):
        ids0 = {variable.id for variable in self.edge_variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this edge.")

    def __repr__(self):
        return f"({self.name}\n{self.constraints=}"

    def __str__(self):
        return self.__repr__()


class GraphOfConvexSets:

    def __init__(self, convex_relaxation=False):
        self.graph = nx.DiGraph()
        self.convex_relaxation = convex_relaxation

    @property
    def vertices(self):
        return list(self.graph.nodes)

    @property
    def edges(self):
        return list(nx.get_edge_attributes(self.graph, "prog").values())

    def add_vertex(self, name=""):
        vertex = Vertex(name, self.convex_relaxation)
        self.graph.add_node(vertex)
        return vertex

    def append_vertex(self, vertex):
        self.graph.add_node(vertex)

    def add_edge(self, tail, head):
        edge = Edge(tail, head, self.convex_relaxation)
        self.graph.add_edge(tail, head, prog=edge)
        return edge

    def append_edge(self, edge):
        self.graph.add_edge(edge.tail, edge.head, prog=edge)

    def remove_vertex(self, vertex):
        """
        remove the vertex and all associated edges
        """
        self.graph.remove_node(vertex)

    def remove_edge(self, edge):
        self.graph.remove_edge(edge.tail, edge.head)

    def add_subgraph(self, gcs):
        self.graph = nx.compose(self.graph, gcs.graph)

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
