import matplotlib.pyplot as plt
import random as rd
import cvxpy as cp
from gcspy import GraphOfConvexSets


class Cell:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {"N": True, "S": True, "E": True, "W": True}

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, wall):
        self.walls[wall] = False


class Maze:
    directions = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1)}

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.cells = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def get_cell(self, x, y):
        return self.cells[x][y]

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.axis("off")
        ax.plot([0, self.nx - 1], [self.ny, self.ny], c="k")
        ax.plot([self.nx, self.nx], [0, self.ny], c="k")
        for x in range(self.nx):
            for y in range(self.ny):
                if self.get_cell(x, y).walls["S"] and (x != 0 or y != 0):
                    ax.plot([x, x + 1], [y, y], c="k")
                if self.get_cell(x, y).walls["W"]:
                    ax.plot([x, x], [y, y + 1], c="k")
        return ax

    def unexplored_neighbors(self, cell):
        neighbours = []
        for direction, (dx, dy) in self.directions.items():
            x2 = cell.x + dx
            y2 = cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.get_cell(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self, seed=0):
        rd.seed(seed)
        n = self.nx * self.ny
        cell_stack = [self.get_cell(0, 0)]
        while len(cell_stack) > 0:
            neighbours = self.unexplored_neighbors(cell_stack[-1])
            if not neighbours:
                cell_stack.pop()
            else:
                direction, next_cell = rd.choice(neighbours)
                self.knock_down_wall(cell_stack[-1], direction)
                cell_stack.append(next_cell)

    def knock_down_wall(self, cell, wall):
        cell.knock_down_wall(wall)
        dx, dy = self.directions[wall]
        neighbor = self.get_cell(cell.x + dx, cell.y + dy)
        neighbor_wall = {"N": "S", "S": "N", "E": "W", "W": "E"}[wall]
        neighbor.knock_down_wall(neighbor_wall)

    def knock_down_walls(self, n, seed=0):
        rd.seed(seed)
        knock_downs = 0
        while knock_downs < n:
            x = rd.randint(1, self.nx - 2)
            y = rd.randint(1, self.ny - 2)
            cell = self.get_cell(x, y)
            walls = [wall for wall, has_wall in cell.walls.items() if has_wall]
            if len(walls) > 0:
                wall = rd.choice(walls)
                self.knock_down_wall(cell, wall)
                knock_downs += 1


def make_maze_gcs(maze_size, knock_downs, seed=0, convex_relaxation=False):
    maze = Maze(maze_size, maze_size)
    maze.make_maze(seed=seed)
    maze.knock_down_walls(knock_downs)
    gcs = GraphOfConvexSets(convex_relaxation=convex_relaxation)

    start = [0.5, 0]
    goal = [maze_size - 0.5, maze_size]
    for i in range(maze_size):
        for j in range(maze_size):
            v = gcs.add_vertex(f"v{(i, j)}")
            x1 = v.add_variable(2)
            x2 = v.add_variable(2)
            v.add_cost(cp.norm(x2 - x1, 2))
            if i == 0 and j == 0:
                v.add_constraint(x1 == start)
            else:
                v.add_constraint(x1 >= [i, j])
                v.add_constraint(x1 <= [i + 1, j + 1])
            if i == maze_size - 1 and j == maze_size - 1:
                v.add_constraint(x2 == goal)
            else:
                v.add_constraint(x2 >= [i, j])
                v.add_constraint(x2 <= [i + 1, j + 1])

    for i in range(maze_size):
        for j in range(maze_size):
            cell = maze.get_cell(i, j)
            v = gcs.get_vertex_by_name(f"v{(i, j)}")
            for direction, (di, dj) in maze.directions.items():
                if not cell.walls[direction]:
                    name = f"v{(i + di, j + dj)}"
                    w = gcs.get_vertex_by_name(name)
                    e = gcs.add_edge(v, w)
                    e.add_constraint(v.variables[1] == w.variables[0])

    s = gcs.get_vertex_by_name(f"v{(0, 0)}")
    t = gcs.get_vertex_by_name(f"v{(maze_size - 1, maze_size - 1)}")
    return gcs, s, t, maze


if __name__ == "__main__":
    maze_size = 10
    knock_downs = 5
    gcs, s, t, maze = make_maze_gcs(maze_size, knock_downs, seed=0)
    ax = maze.plot()
    print("Problem status:", gcs.solve_shortest_path(s, t, verbose=True))
    import numpy as np

    for vertex in gcs.vertices:
        if vertex.y.value is not None and vertex.y.value > 0.5:
            x1, x2 = vertex.variables
            values = np.array([x1.value, x2.value]).T
            ax.plot(*values, c="b", linestyle="--")
    plt.show()
