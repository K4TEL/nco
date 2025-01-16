import numpy as np
import tsplib95
import operator
from aco import ACO, Graph
import matplotlib.pyplot as plt


file_1 = "/lnet/work/people/lutsai/nco/bays29.tsp"
file_2 = "/lnet/work/people/lutsai/nco/berlin52.tsp"

sol_1 = "/lnet/work/people/lutsai/nco/bays29.opt.tour"
sol_2 = "/lnet/work/people/lutsai/nco/berlin52.opt.tour"


iterations = 5  # per setup

gener = [200, 100, 50]
strateg = [2, 1, 0]
a = [0.1, 0.5, 0.8, 1.0, 1.5]  # pheromone importance
b = [2.5, 5.0, 10.0, 15.0]  # heuristic info relative importance
r = [0.25, 0.5, 0.75]  # pheromone residual coeff
qs = [3, 5, 7, 10, 15]  # pheromone intensity
ant_cnt = [10, 25, 50, 75]  # ants number

results = []


def load_problem(input_file):
    problem = tsplib95.load(input_file)
    # print(f"Nodes: {len(list(problem.get_nodes()))}")
    # print(f"Edges: {len(list(problem.get_edges()))}")
    fields = problem.as_name_dict().keys()
    graph = problem.get_graph()
    return problem, graph, fields


def plot(points, path: list, solution: list):
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    y = list(map(operator.sub, [max(y) for i in range(len(points))], y))
    plt.plot(x, y, 'co')

    for n in range(len(solution)-1):
        i = solution[n]-1
        j = solution[n+1]-1
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='y', length_includes_head=True, lw=2)

    for n in range(len(path)-1):
        i = path[n]-1
        j = path[n+1]-1
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True, lw=1, ls="--")

    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.title("Optimal and found solutions on the graph map")
    plt.show()


def solve_tsp_by_aco(data_file, solution_file, ac=10, gen=100, alpha=1.0, beta=10.0, rho=0.5, q=10, strategy=2):
    problem, graph, keywords = load_problem(data_file)
    solution = tsplib95.load(solution_file).tours[0]

    rank = problem.dimension
    if "node_coords" in keywords:
        coords = [problem.node_coords[i] for i in list(problem.get_nodes())]
    else:
        coords = [problem.display_data[i] for i in list(problem.get_nodes())]

    sol_cost = 0
    for n in range(len(solution) - 1):
        from_n = solution[n]
        to_n = solution[n+1]
        sol_cost += problem.get_weight(from_n, to_n)

    cost_matrix = []
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(graph.edges[i+1, j+1]["weight"])
        cost_matrix.append(row)

    aco = ACO(ant_count=ac,
              generations=gen,
              alpha=alpha,  # pheromone importance
              beta=beta,  # heuristic info relative importance
              rho=rho,  # pheromone residual coeff
              q=q,  # pheromone intensity
              strategy=strategy)
    G = Graph(cost_matrix, rank)
    path, cost = aco.solve(G)
    path = [n + 1 for n in path]

    path_start = path.index(1)
    path = path[path_start:] + path[:path_start]

    print('Cost: {} / {} \tpath: {}'.format(int(cost), sol_cost, path))
    # print('Opt: {}, \tpath: {}'.format(sol_cost, solution))
    # plot(coords, path, solution)

    return cost, sol_cost


# solve_tsp_by_aco(file_2, sol_2)
#
# solve_tsp_by_aco(file_1, sol_1)

solution_1 = tsplib95.load(sol_1).tours[0]
solution_2 = tsplib95.load(sol_2).tours[0]

sol_cost_1, sol_cost_2 = 0, 0

best_c1, best_c2 = 1000000, 1000000

for ge in gener:
    for st in strateg:
        for ac in ant_cnt:
            for al in a:
                for be in b:
                    for ro in r:
                        for q in qs:
                            total_c1 = 0
                            total_c2 = 0
                            print(f"Ants: {ac}\t Gen: {ge}\t Alpha: {al}\t Beta: {be}\t Rho: {ro}\t Q: {q}\t Strat: {st}")
                            for i in range(iterations):
                                c_1, sc_1 = solve_tsp_by_aco(file_1, sol_1, ac, ge, al, be, ro, q, st)
                                c_2, sc_2 = solve_tsp_by_aco(file_2, sol_2, ac, ge, al, be, ro, q, st)

                                if c_1 < best_c1:
                                    best_c1 = c_1
                                    results.append((ac, ge, al, be, ro, q, st, c_1, sc_1, c_2, sc_2))
                                elif c_2 < best_c2:
                                    best_c2 = c_2
                                    results.append((ac, ge, al, be, ro, q, st, c_1, sc_1, c_2, sc_2))

                                if sol_cost_1 == 0:
                                    sol_cost_1 = sc_1
                                    sol_cost_2 = sc_2

                                total_c1 += c_1
                                total_c2 += c_2

                            avg_c1 = total_c1 / iterations
                            avg_c2 = total_c2 / iterations

                            print(f"Ants: {ac}\t Gen: {ge}\t Alpha: {al}\t Beta: {be}\t Rho: {ro}\t Q: {q}\t Strat: {st}")

                            print('Opt: {} \tpath: {}'.format(sol_cost_1, solution_1))
                            print('Avg: {}'.format(avg_c1))

                            print('Opt: {} \tpath: {}'.format(sol_cost_2, solution_2))
                            print('Avg: {}'.format(avg_c2))

                            results.append((ac, ge, al, be, ro, q, st, avg_c1, sc_1, avg_c2, sc_2))

                            res = np.array(results)
                            np.savetxt("temp_aco.tsv", res, delimiter="\t", fmt="%s")


res = np.array(results)
np.savetxt("results_aco.tsv", res, delimiter="\t", fmt="%s")
