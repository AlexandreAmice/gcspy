from gcspy import GraphOfConvexSets
import numpy as np
import cvxpy as cp


def make_movie_clustering_problem(n_movies, n_categories, n_clusters, scores=None):
    """
    Assign n_movies to n_clusters. The movies are scored in n_categories and assigned to clusters in such a way
    that each the distance of each movie to it's cluster is minimized.

    Solve this problem as a facility location problem.
    """
    if scores is None:
        scores = np.random.rand(n_movies, n_categories)
    assert np.all(np.logical_and(scores >= 0, scores <= 1))

    gcs = GraphOfConvexSets()

    clusters = []
    for i in range(n_clusters):
        v = gcs.add_vertex(f"cluster{i}")
        x = v.add_variable(n_categories)
        v.add_constraint(x >= 0)
        v.add_constraint(x <= 1)
        clusters.append(v)

    movies = []
    for i, score in enumerate(scores):
        v = gcs.add_vertex(f"movie{i}")
        x = v.add_variable(n_categories)
        v.add_constraint(x == score)
        movies.append(v)

    for cluster in clusters:
        for movie in movies:
            edge = gcs.add_edge(cluster, movie)
            edge.add_cost(cp.sum_squares(cluster.variables[0] - movie.variables[0]))

    return gcs
