import numpy as np
import tsplib95
from lab1_ACO.aco import ACO, Graph


a = [0.1, 0.5, 0,8., 1.0, 1.5]  # pheromone importance
b = [2.5, 5.0, 10.0, 15.0]  # heuristic info relative importance
r = [0.25, 0.5, 0.75]  # pheromone residual coeff
qs = [3, 5, 7, 10, 15]  # pheromone intensity
ant_cnt = [10, 25, 50, 75]  # ants number

lower_att = [np.min(att) for att in [a, b, r, qs, ant_cnt]]
upper_att = [np.max(att) for att in [a, b, r, qs, ant_cnt]]


def load_problem(input_file):
    problem = tsplib95.load(input_file)
    fields = problem.as_name_dict().keys()
    graph = problem.get_graph()
    return problem, graph, fields


class Environment:
    def __init__(self, data_file, solution_file, gen, strategy):
        self.data_file = data_file
        self.solution_file = solution_file
        self.gen = gen
        self.strategy = strategy

        problem, graph, keywords = load_problem(data_file)
        solution = tsplib95.load(solution_file).tours[0]

        self.cities = len(set(solution))

        rank = problem.dimension
        if "node_coords" in keywords:
            coords = [problem.node_coords[i] for i in list(problem.get_nodes())]
        else:
            coords = [problem.display_data[i] for i in list(problem.get_nodes())]

        sol_cost = 0
        for n in range(len(solution) - 1):
            from_n = solution[n]
            to_n = solution[n + 1]
            sol_cost += problem.get_weight(from_n, to_n)

        cost_matrix = []
        for i in range(rank):
            row = []
            for j in range(rank):
                row.append(graph.edges[i + 1, j + 1]["weight"])
            cost_matrix.append(row)

        self.cost_matrix = cost_matrix
        self.rank = rank
        self.opt_cost = sol_cost

        print(f"Environment created with data file: {data_file} and solution file: {solution_file}")

    def func_objective(self, x):
        alpha, beta, rho, q, ac = x

        ac, q = int(ac), int(q)

        aco = ACO(ant_count=ac,
                  generations=self.gen,
                  alpha=alpha,  # pheromone importance
                  beta=beta,  # heuristic info relative importance
                  rho=rho,  # pheromone residual coeff
                  q=q,  # pheromone intensity
                  strategy=self.strategy)
        G = Graph(self.cost_matrix, self.rank)
        path, cost = aco.solve(G)
        path = [n + 1 for n in path]

        path_start = path.index(1)
        path = path[path_start:] + path[:path_start]

        diff_cost = max(0, (cost - self.opt_cost))

        # print(self.cities, len(set(path)))
        if len(set(path)) < self.cities:
            diff_cost += 1000
            cost += 1000

        # cost, opt_cost = solve_tsp_by_aco(self.data_file, self.solution_file, ac=ac,
        #                                   gen=self.gen, alpha=alpha, beta=beta,
        #                                   rho=rho, q=q, strategy=self.strategy)
        print(f"Ants: {ac}\t Gen: {self.gen}\t Alpha: {alpha}\t Beta: {beta}\t Rho: {rho}\t Q: {q}\t Strat: {self.strategy}")
        print('Cost: {} / {} \tDiff: {} \tpath: {}'.format(int(cost), self.opt_cost, diff_cost, path))
        return diff_cost

    def constrains(self, x):
        a, b, r, qs, ant_cnt = x

        a_diff = a - upper_att[0] if a > upper_att[0] else max(0, lower_att[0] - a)
        b_diff = b - upper_att[1] if b > upper_att[1] else max(0, lower_att[1] - b)
        r_diff = r - upper_att[2] if r > upper_att[2] else max(0, lower_att[2] - r)
        qs_diff = qs - upper_att[3] if qs > upper_att[3] else max(0, lower_att[3] - qs)
        ac_diff = ant_cnt - upper_att[4] if ant_cnt > upper_att[4] else max(0, lower_att[4] - ant_cnt)

        return sum([a_diff, b_diff, r_diff, qs_diff, ac_diff])

