from particle import *
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import time
from objective import *
import random
from random import uniform
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from particle import particle_single
from objective import Environment

from lab1_ACO.aco import ACO, Graph

from pathos.multiprocessing import ProcessPool as Pool



class pso:
    '''
    c_param -> personal parameter
    s_param -> sozial parameter
    v_weight -> weight for old velocity
    swarm -> all particles
    comp_swarm -> swarm + g_best + p_best
    l_bound -> lower bounds
    u_bound -> upper bounds
    integer -> Integer constraint for every attribute
    vmax -> max velocity
    g_best -> best particle of the swarm
    multi -> multi objective or single objective
    '''
    
    def __init__(self,att,l_b,u_b,obj_func,constraints=[],c=2.1304,s=1.0575,w=0.4091,pop=156,vm=np.nan,integer=False):
        print(f"Initializing PSO with {pop} particles")
        if np.isnan(vm):
            vm = np.array([u_b[i]-l_b[i] for i in range(att)])
        if type(vm) != np.ndarray and type(vm) != list:
            vm = np.array([vm for i in range(att)])
        if len(vm)!=att:
            np.append(vm, [vm[len(vm)-1]for i in range(len(l_b),att)])
        if type(integer) != list:
            integer = np.array([integer for i in range(att)])
        self.c_param = c
        self.s_param = s
        self.v_weight = w
        self.l_bound = l_b
        self.u_bound = u_b
        self.integer = integer
        self.vmax = vm

        def initialize_particle(args):
            obj_func, att, constraints, vm, l_b, u_b, integer = args
            return particle_single(obj_func, att, constraints, vm, l_b, u_b, integer)

        if type(obj_func)!=list:
            self.multi = False

            # Parallel initialization of particles
            args_list = [(obj_func, att, constraints, vm, l_b, u_b, integer) for _ in range(pop)]
            with Pool(ncpus=mp.cpu_count()) as pool:
                swarm = pool.map(initialize_particle, args_list)
                # while not swarm.ready():
                #     time.sleep(5);
                #     print(".", end=' ')
                #
                # swarm = swarm.get()

            self.swarm = swarm

            # self.swarm = [particle_single(obj_func,att,constraints,vm,l_b,u_b,integer) for i in range(pop)]
        else:
            self.multi = True
            # with ProcessPoolExecutor(max_workers=16, max_tasks_per_child=5) as executor:
            #     self.swarm = list(executor.map(
            #         lambda _: particle_multi(obj_func, att, constraints, vm, l_b, u_b, integer),
            #         range(pop)
            #     ))
            self.swarm = [particle_multi(obj_func,att,constraints,vm,l_b,u_b,integer) for i in range(pop)]
            self.comp_swarm = self.swarm
        if self.multi:
            self.non_dom_sort()
        for part in self.swarm:
            part.init_p_best()
        self.set_g_best()

        print(f"PSO initialized with {pop} particles, lowest diff from optimal solution:\t{self.g_best.get_obj_value()}")
        
    def __repr__(self):
        if self.multi:
            return f" The multi objective particle swarm optimizer, with {len(self.swarm)} particles, {len(self.swarm[0].position)} Attributes, {len(self.swarm[0].obj_functions)} objective functions"
        else:
            return f" The single objective particle swarm optimizer, with {len(self.swarm)} particles, {len(self.swarm[0].position)} Attributes"
        
    def non_dom_sort(self):
        # fast non domination sort
        F = []
        F1 = []
        for p in self.comp_swarm:
            Sp = []
            n_p = 0
            for q in self.comp_swarm:
                if p.dominates(q):
                    Sp.append(q)
                elif q.dominates(p):
                    n_p +=1
            p.S = Sp
            p.n = n_p
            if n_p==0:
                F1.append(p)
                p.rank = 0
        F.append(F1)
        i=0
        while F[i]:
            H = []
            for p in F[i]:
                for q in p.S:
                    q.n -= 1
                    if q.n == 0:
                        H.append(q)
                        q.rank = i+1
            i +=1
            F.append(H)
        F.pop()
        # crowding distance
        for Fi in F:
            l = len(Fi)
            for parti in Fi:
                parti.distance = 0
            for m in range(len(Fi[0].obj_functions)):
                m_obj = [x.obj_values[m] for x in Fi]
                if l>1:
                    Fi_sorted = [Fi[j] for j in np.argsort(m_obj)]
                else:
                    Fi_sorted = Fi
                Fi_sorted[0].distance = np.inf
                Fi_sorted[-1].distance = np.inf
                for i in range(2,l-1):
                    Fi_sorted[i].distance = Fi_sorted[i].distance + Fi_sorted[i+1].obj_values[m] - Fi_sorted[i-1].obj_values[m]
                    
    def set_g_best(self):
        self.g_best = self.swarm[0]
        for part in self.swarm[1:]:
            if part.compare(self.g_best):
                self.g_best = copy.deepcopy(part)
        
    def moving(self,steps, time_termination):
        t0 = time.time()
        for i in range(steps):
            print(f"\tStep {i+1}/{steps}")
            print(f"\tBest solution error: {self.g_best.get_obj_value()}")
            if time_termination != -1 and time.time()-t0 > time_termination:
                break
            if self.multi:
                # comp_swarm needed to update rank and distance of p_best und g_best
                # first is global best ; the part and p_best alternating
                #if self.multi:
                self.comp_swarm=[]
                self.comp_swarm.append(self.g_best)
            for part in self.swarm:
                #calc new velocity
                r1 = random.random()
                r2 = random.random()
                new_v = self.v_weight*part.velocity + self.c_param*r1*(part.best_p.position-part.position) + self.s_param*r2*(self.g_best.position-part.position)
                # control vmax
                new_v = np.array([new_v[i] if new_v[i]>-self.vmax[i] else -self.vmax[i] for i in range(len(new_v))])
                new_v = np.array([new_v[i] if new_v[i]<self.vmax[i] else self.vmax[i] for i in range(len(new_v))])                
                
                #calc new position
                new_p = part.position + new_v
                for i in range(len(new_p)):
                    if self.integer[i]:
                        new_p[i] = int(new_p[i])
                
                # stick to bound
                new_p = np.array([new_p[i] if new_p[i]>self.l_bound[i] else self.u_bound[i] - abs(self.l_bound[i]-new_p[i])%(self.u_bound[i]-self.l_bound[i]) for i in range(len(new_p))])
                new_p = np.array([new_p[i] if new_p[i]<self.u_bound[i] else self.l_bound[i] + abs(self.u_bound[i]-new_p[i])%(self.u_bound[i]-self.l_bound[i]) for i in range(len(new_p))])               
                
                part.set_velocity(new_v)
                part.set_position(new_p)
                
                #add to comp_swarm
                if self.multi:
                    self.comp_swarm.append(part)
                    self.comp_swarm.append(part.best_p)
                
            if self.multi:
                self.non_dom_sort()
                # set g_best with new rank and distance
                self.g_best = copy.deepcopy(self.comp_swarm[0])
                j=1
                new_swarm = self.comp_swarm[1:-1:2]
                self.swarm = copy.deepcopy(new_swarm)
                j+=1
                
            for part in self.swarm:
                if self.multi:
                    #set swarm and p_best with new rank and distance 
                    part.best_p = copy.deepcopy(self.comp_swarm[j])
                    j+=2
                #compare part with g_best
                if part.compare(self.g_best):
                    self.g_best = copy.deepcopy(part)
                #update p_best
                part.compare_p_best()

    def get_solution(self,whole_particle=False):
        solution = []
        if self.multi:
            for part in self.swarm:
                if part.rank == 0:
                    if whole_particle:
                        solution.append(part)
                    else:
                        solution.append(part.get_obj_value())
                if part.best_p.rank == 0 and all(part.position!=part.best_p.position):
                    if whole_particle:
                        solution.append(part.best_p)
                    else:
                        solution.append(part.best_p.get_obj_value())
        else:
            if whole_particle:
                solution = self.g_best
            else:
                solution = self.g_best.get_obj_value()
                
        return solution 
