import sys
sys.path.append('../')

from pso import pso
from objective import *

gener = [100, 200, 50]  # generation of sworm
strateg = [2, 1, 0]  # ACO strategy

# a = [0.1, 0.5, 0,8, 1.0, 1.5]  # pheromone importance
# b = [2.5, 5.0, 10.0, 15.0]  # heuristic info relative importance
# r = [0.25, 0.5, 0.75]  # pheromone residual coeff
# qs = [3, 5, 7, 10, 15]  # pheromone intensity
# ant_cnt = [10, 25, 50, 75]  # ants number

population = 15  # sworm size
iterations = 25  # moving steps
N_att = 5  # optimized variables number


file_1 = "/lnet/work/people/lutsai/nco/bays29.tsp"
file_2 = "/lnet/work/people/lutsai/nco/berlin52.tsp"

sol_1 = "/lnet/work/people/lutsai/nco/bays29.opt.tour"
sol_2 = "/lnet/work/people/lutsai/nco/berlin52.opt.tour"

problem_1, _, _ = load_problem(file_1)
problem_2, _, _ = load_problem(file_2)

solution_1 = tsplib95.load(sol_1).tours[0]
solution_2 = tsplib95.load(sol_2).tours[0]

city_1 = set(solution_1)
city_2 = set(solution_2)

sol_cost_1, sol_cost_2 = 0, 0

for n in range(len(solution_1) - 1):
    from_n = solution_1[n]
    to_n = solution_1[n + 1]
    sol_cost_1 += problem_1.get_weight(from_n, to_n)

for n in range(len(solution_2) - 1):
    from_n = solution_2[n]
    to_n = solution_2[n + 1]
    sol_cost_2 += problem_2.get_weight(from_n, to_n)

results = []

for ge in gener:
    for st in strateg:
        env_1 = Environment(file_1, sol_1, ge, st)
        env_2 = Environment(file_2, sol_2, ge, st)

        sworm_1 = pso(N_att, lower_att, upper_att, env_1.func_objective, constraints=env_1.constrains,
                c=2.13, s=1.05, w=0.41, pop=population, integer=[False, False, False, True, True])

        print(f"\tMoving for {iterations} steps")
        sworm_1.moving(iterations, -1)

        solut_1 = sworm_1.get_solution(True)
        alpha, beta, rho, q, ac = solut_1.position
        print(f"\tAnts: {ac}\t Gen: {ge}\t Alpha: {alpha}\t Beta: {beta}\t Rho: {rho}\t Q: {q}\t Strat: {st}")
        print('\tOpt: {} \tpath: {}'.format(sol_cost_1, solution_1))
        print('\tBest: {}'.format(solut_1.obj_value + sol_cost_1))

        sworm_2 = pso(N_att, lower_att, upper_att, env_2.func_objective, constraints=env_2.constrains,
                      c=2.13, s=1.05, w=0.41, pop=population, integer=[False, False, False, True, True])

        print(f"\tMoving for {iterations} steps")
        sworm_2.moving(iterations, -1)

        solut_2 = sworm_2.get_solution(True)
        alpha, beta, rho, q, ac = solut_2.position
        print(f"\tAnts: {ac}\t Gen: {ge}\t Alpha: {alpha}\t Beta: {beta}\t Rho: {rho}\t Q: {q}\t Strat: {st}")
        print('\tOpt: {} \tpath: {}'.format(sol_cost_2, solution_2))
        print('\tBest: {}'.format(solut_2.obj_value + sol_cost_2))

        results.append((ac, ge, alpha, beta, rho, q, st, population, iterations,
                        solut_1.obj_value + sol_cost_1, sol_cost_1, solut_2.obj_value + sol_cost_2, sol_cost_2))

        res = np.array(results)
        res = np.round(res, 3)
        np.savetxt(f"temp_{population}_pso.tsv", res, delimiter="\t", fmt="%s")


res = np.array(results)
res = np.round(res, 3)
np.savetxt("results_pso.tsv", res, delimiter="\t", fmt="%s")
