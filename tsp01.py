#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
TSP solved with local search methods.
Local neighborhood with subtour-reversal (see Intro to OR 14.1)

Tours are implemented as linked lists of Points.
To implement & improve: methods to escape local minima:
    * Tabu search
    * Simulated Annealing
    * Genetic Algorithms
    * ...
"""
from __future__ import division

import math, random, copy, sys
from collections import namedtuple
from matplotlib import pyplot as plt
import pdb

class Point:
    def __init__(self, x, y, name=''):
        self.x = x
        self.y = y
        self.name = name
        self.next = None
    
    def __str__(self):
        #return 'Point %s (%.1f, %.1f)' % (self.name, self.x, self.y)
        #return 'Point %s' % (self.name)
        return str(self.name)
    
    __repr__ = __str__
    
class Tour(list):
    def __init__(self):
        self.len = None
        
    def __str__(self):
        s = ''
        point = self[0]
        start = point
        while point.next != start:
            s += str(point) + ' '
            point = point.next
        return s
    
#    def __cmp__(self, other):
#        if len(self) != len(other):
#            return False
#        return all(self[i] == other[i] for i in range(len(self)))
    
    @property
    def length(self):
        if not self.len:
            self.len = tour_length(self)
        return self.len
    
    def set_length(self, length):
        self.len = length
    
    def check_integrity(self):
        """check if we make a complete tour, including all the points.
        This property should be maintained; this method is just for
        debugging purposes.
        """
        # 1. find a tour
        point = self[0]
        a_tour = [point]
        while point.next not in a_tour:
            a_tour.append(point.next)
            point = point.next
            
        # check if tour is maximal:
        return len(a_tour) == len(self)
    
    def edges(self):
        """returns ordered list of edges."""
        edges = []
        start = self[0]
        point = start
        while point.next is not start:
            edges.append((point, point.next))
            point = point.next
        edges.append((point, start)) # add last edge, from last to first point
        return edges
            

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), name=str(i)))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def read_input(input_data, max_count=None):
        # parse the input
    lines = input_data.split('\n')

    n_count = int(lines[0])
    if max_count: n_count = max_count

    points = Tour()
    for i in range(1, n_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), name=str(i)))
    return n_count, points

def trivial(points):
    for i in range(len(points)-1):
        points[i].next = points[i+1]
    points[-1].next = points[0]
    return points

def greedy(tour):
    """TSP tour obtained by always choosing the best next Point."""
    new_tour = Tour()
    point = tour[0]
    while len(new_tour) < len(tour):
        next = sorted([(length(point, p), p) for p in tour if p not in new_tour and p != point])[0][1]
        point.next = next
        point = next
        new_tour.append(point)
    new_tour[-1].next = new_tour[0]
    return new_tour

def plot_tour(tour, figsize=(5,5), show_names=False):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    for point in tour:
        ax1.plot(point.x, point.y, 'bo')
        ax1.plot([point.x, point.next.x] ,
                 [point.y, point.next.y], 'b-')
        if show_names: # plots the name (number) of the node
            ax1.text(point.x, point.y, point.name)
            
    plt.title('Length = %.2f' % tour_length(tour))
    plt.show()

def tour_length(points):
    return sum(length(point, point.next) for point in points)

def opt_2(tour):
    """Defines local neighborhood around proposal solution.
    Here defined as all the possible reversed subtours.
    This is an implementation of the Sub-Tour Reversal Algorithm
    as in Intro to OR - 14.1
    """
    nbh = []
    tl = tour.length
    for i in range(len(tour)):
        for j in range(i+1, len(tour)): # pick any two edges
            new_tour = copy.deepcopy(tour)
            p1, p2 = new_tour[i], new_tour[i].next
            p3, p4 = new_tour[j], new_tour[j].next
            new_tl = make_subtour_reversal(new_tour, p1, p2, p3, p4)
            new_tour.len = new_tl
            nbh.append((new_tl, new_tour))
    return sorted(nbh)

def opt_3(tour):
    """Local neighbourhoud, by swapping 3 edges."""
    nbh = []
    edges = tour.edges()
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            for k in range(j+1, len(edges)): # pick any three edges
              new_tour = copy.deepcopy(tour) # make copy of BOTH (i.e. .deepcopy) tour object, and internal points
              new_tl = make_subtour_reversal_opt3(new_tour, i, j, k)
              nbh.append((new_tl, new_tour))
    return sorted(nbh)

def one_opt_2(tour):
    """Get one, randomly sampled, opt-2 neighbor of tour.
    Implemented for use with simulated anealing"""
    edges = tour.edges()
    i = random.randint(0, len(edges)-2)
    j = random.randint(i+1, len(edges)-1)
    new_tour = copy.deepcopy(tour) # make copy of BOTH (i.e. .deepcopy) tour object, and internal points
    p1, p2 = new_tour[i], new_tour[i].next
    p3, p4 = new_tour[j], new_tour[j].next
    new_tl = make_subtour_reversal(new_tour, p1, p2, p3, p4)
    new_tour.len = new_tl
    return (new_tl, new_tour)

def one_opt_3(tour):
    """Get one, randomly sampled, opt-3 neighbor of tour.
    Implemented for use with simulated anealing"""
    edges = tour.edges()
    i = random.randint(0, len(edges)-3)
    j = random.randint(i+1, len(edges)-2)
    k = random.randint(j+1, len(edges)-1)
    new_tour = copy.deepcopy(tour) # make copy of BOTH (i.e. .deepcopy) tour object, and internal points
    new_tl = make_subtour_reversal_opt3(new_tour, i, j, k)
    return (new_tl, new_tour)

def make_subtour_reversal_opt3(new_tour, i, j, k):
    """In-place creation of new_tour with reversed subtour.
    OPT-3 version"""
    edges = new_tour.edges()
    p1, p2 = edges[i]
    p3, p4 = edges[j]  
    p5, p6 = edges[k]
    new_tl  = new_tour.length - length(p1, p2) - length(p3, p4) - length(p5, p6)
    new_tl += length(p1, p5) + length(p4, p2) + length(p3, p6)
    subtour = (p4, p5)
    reverse(subtour)
    p1.next = p5
    p4.next = p2
    p3.next = p6
    new_tour.set_length(new_tl)
    if not new_tour.check_integrity():
        print 'Inconsistency tour created'
        pdb.set_trace()
    return new_tl
    

def make_subtour_reversal(new_tour, p1, p2, p3, p4):
    """In-place creation of new_tour with reversed subtour."""
    new_tl  = new_tour.length - length(p1, p2) - length(p3, p4)
    new_tl += length(p1, p3) + length(p2, p4)
    subtour = (p2, p3)
    reverse(subtour)
    p1.next = p3
    p2.next = p4
    return new_tl
    
def reverse(subtour):
    """Routine that reverse the subtour in place.
    input: subtour = (p1, p2), where p1 = starting Point
        and p2 = end Point of subtour.
    """
    start, stop = subtour
    p = start
    reversed = []
    while p is not stop: # get all points in subtour
        reversed.append(p)
        p = p.next
    reversed.append(stop)
    
    prev = start
    for p in reversed[1:]: # reverse the pointers to next
        p.next = prev
        prev = p
    

def hill_climbing(initial, max_iter=1000, neighborhood=opt_2):
    """Find local minimum by always taking a step to best neighbor.
    How neighborhood is defined depends on problem.
    """
    tour = initial
    best_val = tour_length(tour)
    n_iter = 0
    while n_iter < max_iter:
        nbh = neighborhood(tour)
        if abs(nbh[0][0] - best_val) < 10**-7:
            print 'Local minimum found after %d iterations.' % n_iter 
            return best_sol
        else:
            print 'iter %d -> len(nbh) = %d, best = %.2f (true: %.2f)' \
                % (n_iter, len(nbh), nbh[0][0], tour_length(nbh[0][1]))
            l, tour = nbh[0]
            best_val = l
            best_sol = tour
            if not tour.check_integrity():
                # catch it if tour is not good (tour constraint is violated)
                print 'is_tour: ', tour.check_integrity()
        n_iter += 1
    return best_sol

def tabu_search(initial, max_iter=1000, lifetime=5, neighborhood=opt_2):
    """Find local minimum by always taking a step to best neighbor.
    How neighborhood is defined depends on problem.
    Tour is saved in tabu list under it's string form.
    """
    tabu_list = {}
    current_tour = initial
    best_val = current_tour.length
    n_iter = 0
    while n_iter < max_iter:
        nbh = neighborhood(current_tour)

        # filter neighorhood based on tabu_list
        nbh = [(l, tour) for l, tour in nbh if str(tour) not in tabu_list.keys()]
        
        # keep best available solution in filtered neighborhood
        current_val, current_tour = nbh[0] 

        # if this solution is best, keep reference
        if current_val < best_val:
            best_val = current_val
            best_tour = current_tour
        
        # add this tour to the list
        tabu_list[str(current_tour)] = n_iter
        
        # remove 'old' tours from tabu_list
        for t in tabu_list.keys():
            if tabu_list[t] < n_iter - lifetime:
                tabu_list.pop(t)
        
        print 'iter %d -> best = %.2f, current = %.2f [|nbh| = %d]' % (n_iter, best_val,
                                                          current_val, len(nbh))
        n_iter += 1
    return best_tour

def schedule01(T_init=1000.0):
    T = T_init
    n_calls = 0
    while True:
        n_calls += 1
        if n_calls <= 20:
            pass
        elif 20 < n_calls <= 500:
            T = T*.995
        elif 500 < n_calls <= 1000:
            T = T*.995            
        else:
            pass # keep T as is
        yield T

def simulated_annealing(initial, max_iter=1000, T_init=1000,
                        neighborhood=one_opt_3, schedule=schedule01):
    """
    Find a minimum of the objective function, using simulated annealing (Intro to OR 14.3)
    """
    current_tour = initial
    best_val = tour_length(tour)
    best_tour = current_tour
    temp_sched = schedule(T_init)
    
    for n_iter in range(max_iter):
        # Adapt temperature
        T = temp_sched.next()
    
        # select a move at random
        current_val, candidate = neighborhood(current_tour)

        if current_val < best_val:
            # accept candidate, and set new best tour
            print '1. iter %d -> accepted: value = %.2f [T=%.2f, best=%.2f]' \
                % (n_iter, current_val, T, best_val)
            best_val = current_val
            best_tour = candidate
            current_tour = candidate

        elif random.random() < math.exp((best_val-current_val)/T):
            # accept candidate for next current tour
            print '2. iter %d -> accepted: value = %.2f [T=%.2f, best=%.2f]' \
                % (n_iter, current_val, T, best_val)
            current_tour = candidate

        else:
            # refuse candidate
            print '3. iter %d -> refused: value = %.2f [T=%.2f, best=%.2f]' \
                % (n_iter, current_val, T, best_val)

    return best_tour
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
    else:
        file_location = './data/tsp_51_1'
        # './data/tsp_51_1' -> BEST: OPT_2: 435.09;         OPT_3: 434.82
        #       with tabu search:    OPT_2: 434.11 (opt?);  OPT_3: 429.48
        #       with simulated annealing:    OPT_2:       ; OPT_3: 
        # './data/tsp_70_1' -> BEST: OPT_2: 691.08; OPT_3: 677.19 (15 iters)
        # './data/tsp_76_1' -> BEST: OPT_2: 565.73; OPT_3: 
    with open(file_location, 'r') as  input_data_file:
        input_data = ''.join(input_data_file.readlines())
    n_points, points = read_input(input_data, max_count=15)
    tour = greedy(points)
    print tour
    
    print '====================== Hill-climbing ======================'
    new_tour = hill_climbing(tour, max_iter=50, neighborhood=opt_3)
    print '======================  Tabu Search  ======================'
    new_tour = tabu_search(tour, max_iter=20,
                           lifetime=5, neighborhood=opt_2)
    print '==================  Simulated Annealing  =================='                           
#    new_tour = simulated_annealing(tour, max_iter=1000000, T_init=1000,
#                                   neighborhood=one_opt_2)
    
    plot_tour(tour, show_names=True)
    plot_tour(new_tour, show_names=True)
    

