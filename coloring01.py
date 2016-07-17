# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:33:51 2016

coloring03.py

Implement constraint propagation with AC-3.
    'Maintaining Arc Consistency (MAC)' -> AIMA 218
and backtracking

To implement:
    * backmarking
    * backjumping (backtrack to most recent assignment in the conflict set)
    * dynamic reordering: OK
    * other methods from Discrete Optimization
    * local search for constraint satisfaction problems
        * An elementary form of hill-climbing has been integrated (08 Jul 16)
            Its neighborhood looks for the worst node (most violations),
            and then finds new postion that improves it the most
        * This method is almost guaranteed to get stuck in local minimum where
            the number of violations > 0 (and thus not a feasible solution)
        * TO OD: implement tabu search / simulated anealing to find global optimum
"""

from __future__ import division
import sys, copy, time

import pdb

def timeit(fn):
    def wrapper(*args):
        start = time.time()
        result = fn(*args)
        print '%12s -> %.5f' % (fn.__name__, time.time()-start)
        return result
    return wrapper

def parse_input(input_data):
    """
    input: input_data as found in the inout file.
    output: graph dict, representing the graph as adjacency list
    """
	# parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    
    # graph is implemented as an adjacency list, with numbers representing the nodes
    graph = {}
    for node1, node2 in edges:
		if node1 not in graph:
			graph[node1] = [node2]
		else:
			graph[node1].append(node2)
		if node2 not in graph:
			graph[node2] = [node1]
		else:
			graph[node2].append(node1)
    return graph

class bookkeeping:
    length_queue = 0
    n_nodes = 0
    
class CSP:
    """
    Implements a container for a constraint satisfaction problem.
    Only works with binary constraints (different)
    attributes:
        X: list of variables in the problem
        D: dict of domains for each variable
        C: dict of constraints to satifiy for each variable.
        graph: the enitire graph (implemented as an adjacency list)
    """
    def __init__(self, graph, colors):
        self.X = graph.keys()
        self.D = colors
        self.C = []
        for node in graph:
            for node2 in graph[node]:
                self.C.append((node, node2))
        self.graph = graph
    
    def domain(self, x):
        if x in self.D:
            return self.D[x]
        else:
            raise ValueError('Variable %s not present in CSP' % str(x))
    
    def neighbours(self, x):
        if x in self.graph:
            return self.graph[x]
        else:
            raise ValueError('Variable %s not present in CSP' % str(x))

class Config:
    """Simple container class for both graph structure and color assignment.
    Used in local search functions"""
    def __init__(self, graph, n_colors=1):
        self.graph = graph
        self.colors = dict()
        for node in graph:
            self.colors[node] = n_colors - 1
    
    def nodes(self):
        return self.graph.keys()
    
    def n_colors(self):
        return max(self.colors.values()) + 1
    
    def __str__(self):
        s = ''
        for x in sorted(self.nodes()):
            s += str(self.colors[x]) + ' '
        return s
    
def solved(csp):
    """Returns True is csp is solved, meaning each domain has been reduced
    to size 1."""
    return all(len(csp.D[d]) == 1 for d in csp.D.keys())

def first_open(csp):
    """Returns first variable with len(domain) > 0."""
    for x in csp.X:
        if len(csp.domain(x)) > 1:
            return x

def least_options(csp):
    """Returns first variable smallest len(domain) [> 1]."""
    least_len, best_node = float('inf'), None
    for x in csp.X:
        if len(csp.domain(x)) > 1 and len(csp.domain(x)) < least_len:
            best_node = x
            least_len = len(csp.domain(x))
    return best_node

def AC3(csp):
    """Arc-consistency algorithm AC-3.
    Returns False is an inconsistency is found,
    and True otherwise.
    inputs: csp, a binary CSP with components (X, D, C)
    local vars: queue, a queue of arcs, initially
                  all the the arcs in csp
    From AIMA - p.209
    """
    queue = csp.C[:]
    while len(queue) > 0:
        #pdb.set_trace()
        Xi, Xj = queue.pop(0) # remove first arc
        if revise(csp, Xi, Xj):
            if len(csp.domain(Xi)) == 0: # inconsistency
                return False
            for Xk in csp.neighbours(Xi):
                if Xk != Xj:
                    queue.append((Xk, Xi))
    return True

def revise(csp, Xi, Xj):
    """returns True iff we revise the domain of Xi"""
    #pdb.set_trace()
    revised = False
    for x in csp.domain(Xi): # now check if dom(Xj) contains no other values than x, then remove x from dom(Xi)
        if len(csp.domain(Xj)) == 1 and x in csp.domain(Xj):
            csp.D[Xi].remove(x)
            revised = True
    return revised

@timeit
def BFS(csp, order=None):
    if order == None:
        order = csp.graph.keys()
    queue = [csp]
    while len(queue) > 0:
        csp = queue.pop()
        if solved(csp): return csp # return when solution is found
        #X = first_open(csp) # find first node with non-singular domain
        X = least_options(csp) # find node with least options -> dynamic re-ordering
        for choice in csp.domain(X):
            # BOOKKEEPING
            bookkeeping.n_nodes += 1
            new_csp = copy.deepcopy(csp)
            new_csp.D[X] = [choice]
            if AC3(new_csp):
                queue.append(new_csp)
                # BOOKKEEPING
                if len(queue) > bookkeeping.length_queue:
                    bookkeeping.length_queue = len(queue)
    return False
    
def create_output(n_colors, csp):
    if isinstance(csp, CSP):
        s = ''
        s += str(n_colors) + ' 1\n' # 0/1 -> optimality (not) proven
        for x in sorted(csp.X):
            s += str(csp.D[x][0]) + ' '
        return s
    elif isinstance(csp, Config):
        s = ''
        s += str(n_colors) + ' 1\n' # 0/1 -> optimality (not) proven
        for x in sorted(csp.graph.keys()):
            s += str(csp.colors[x]) + ' '
        return s

def check_constraints(csp):
    for node in csp.X:
        for node2 in csp.graph[node]:
            if csp.D[node] == csp.D[node2]:
                return False
    return True

def solve_it(input_data, ):
    graph = parse_input(input_data)
    n_nodes  = len(graph)
    for n_colors in [17]:#range(2, 10):
        colors = dict()
        for node in graph.keys():
            colors[node] = range(n_colors)
        
        csp = CSP(graph, colors)
        result = BFS(csp)
        if isinstance(result, CSP): break
        else: 
            print 'No solution with %d colors.' % n_colors
    print '# colors needed = %d' % n_colors
    return create_output(n_colors, result)

def find_best(config, max_colors):
    """The best next config is defined as follows:
    1. Find node with largest number of violations
    2. For that node, find color that decreases the # of violations the most.
    Change color of this node, and return a copy of the config."""
    new_config = copy.deepcopy(config)
    # 1. Find node with largest number of violations
    worst_node = 0
    most_violations = 0
    for node in new_config.nodes():
        if n_violations(new_config, node) > most_violations:
            most_violations = n_violations(new_config, node)
            worst_node = node
    # 2. For that node, find color that decreases the # of violations the most.
    best_color = 0
    least_violations = float('inf')
    for color in range(max_colors):
        new_config.colors[worst_node] = color
        if n_violations(new_config, worst_node) < least_violations:
            least_violations = n_violations(new_config, worst_node)
            best_color = color
    new_config.colors[worst_node] = best_color
    return new_config

def neighborhood(config, max_colors):
    """The neighborhood is defined as follows:
    For each node, find the config with the least violations."""
    nbh = []
    for node in config.nodes():
        new_config = copy.deepcopy(config)
        best_color = 0
        least_violations   = float('inf')
        initial_violations = n_violations(config, node)
        
        for color in [c for c in range(max_colors) if c != config.colors[node]]: # iterate over all colors, except the current one for node
            new_config.colors[node] = color
            if n_violations(new_config, node) < least_violations:
                least_violations = n_violations(new_config, node)
                best_color = color
        new_config.colors[node] = best_color
        nbh.append((initial_violations - least_violations, 
                    new_config, node))
    return sorted(nbh, reverse=True)


def hill_climbing(config, max_colors, max_iter=100, initial='reduced'):
    """Gradually move to better solution, by establishing sequence
    of feasible solutions.
    See Discrete Optimization lecture-ls-4.pdf
    """
    # n_colors, current_config = greedy(config)
    if initial == 'reduced': # reset all nodes with color in config > max_color
        for node in config.nodes():
            if config.colors[node] > max_colors - 1:
                config.colors[node] = random.randint(0, max_colors - 2)
    elif initial == 'random': # set ALL nodes to random color
        for node in config.nodes():
            config.colors[node] = random.randint(0, max_colors - 2)
            
    n_iter = 0
    while n_iter < max_iter:
        config = find_best(config, max_colors)
        print 'iter %d -> # violations = %d' % (n_iter, total_violations(config))
        n_iter += 1
    return config

def tabu_search(config, max_colors, max_iter=100, max_lifetime=1024):
    """Find an optimum, by escaping local optima by sometimes makes worse moves,
    and don't allow immediate backtracking.
    The time spent on the tabu list is increased dynamically.
        (optimal schedule to be determined)
    It's not logical to exlude entire configurations with the tabu list
        -> focus on individual node? something else? to improve
    """
    tabu_list = {}
    best_config = None
    best_value  = float('inf')
    n_iter = 0
    lifetime = 5
    while n_iter < max_iter:
        nbh = neighborhood(config, max_colors)
        jj = 0
        while True:
            _, config, _ = nbh[jj]
            if str(config) not in tabu_list:
                break
            jj += 1
        if total_violations(config) < best_value:
            best_value  = total_violations(config)
            best_config = config
        #print 'config = ', config
        print 'iter %d -> best = %d, current = %d [lifetime = %d]' \
            % (n_iter, best_value, total_violations(config), lifetime)
        # print 'config = \n', config
        
        if best_value == 0: 
            return best_config
            
        # dynamically increaese lifetime of items of tabu_list
        if best_value == total_violations(config):
            lifetime = min(max_lifetime, lifetime*1.1)
        
        # add current config to tabu_list:
        tabu_list[str(config)] = n_iter
        
        # remove old items from tabu_list
        for c in tabu_list.keys():
            if tabu_list[c] < n_iter - lifetime:
                tabu_list.pop(c)
        
        n_iter += 1
    return best_config

def tabu_search2(config, max_colors, max_iter=100, max_lifetime=1024):
    """Find an optimum, by escaping local optima by sometimes makes worse moves,
    and don't allow immediate backtracking.
    The time spent on the tabu list is increased dynamically.
        (optimal schedule to be determined)
    It's not logical to exlude entire configurations with the tabu list
        this (2nd) implementation focusses on the node that got changed
    """
    tabu_list = {}
    best_config = None
    best_value  = float('inf')
    n_iter = 0
    lifetime = 5
    while n_iter < max_iter:
        nbh = neighborhood(config, max_colors)
        jj = 0
        while True:
            _, config, node = nbh[jj]
            if node not in tabu_list:
                break
            jj += 1
        if total_violations(config) < best_value:
            best_value  = total_violations(config)
            best_config = config
        
        if best_config == config:
            print 'iter %d -> best = %d, current = %d [lifetime = %d]' \
                % (n_iter, best_value, total_violations(config), lifetime)
        
        if best_value == 0: 
            return best_config

        # add current config to tabu_list:
        tabu_list[node] = n_iter
        
        # remove old items from tabu_list
        for n in tabu_list.keys():
            if tabu_list[n] < n_iter - lifetime:
                tabu_list.pop(n)
        
        n_iter += 1
    return best_config

def n_violations(config, node):
    n_viols = 0
    for node2 in config.graph[node]:
        if config.colors[node] == config.colors[node2]:
            n_viols += 1
    return n_viols

def total_violations(config):
    n_viols = 0
    for node in config.nodes():
        n_viols += n_violations(config, node)
    return int(n_viols/2)
    

def greedy(csp):
    """Assign to each node a color, not present in it's neighbors.
    If no color is available because of the constraints, increase the number
    of available colors.
    """
    graph = csp.graph
    colors = dict()
    n_colors = 0
    for node, neighbors in graph.items():
        set = False
        neighbor_colors = [colors[n] for n in neighbors if n in colors] # create the list of colors of the neighbors wich already have been set
        for color in range(n_colors):
            if color not in neighbor_colors: # if color is still available for the node
                colors[node] = color         # ... assign it this node
                set = True
        # if no color was available and the node has not been set:
        if not set:
            colors[node] = n_colors
            n_colors += 1
    
    # set everything in the correct form for output processing
    if isinstance(csp, CSP):
        for node in csp.X:
            csp.D[node] = [colors[node]]
    elif isinstance(csp, Config):
        for node in csp.graph.keys():
            csp.colors[node] = colors[node]
    return n_colors, csp
            
            
            
       
if __name__ == '__main__':    
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
    else:
        file_location = './data/gc_50_3'

    with open(file_location, 'r') as input_data_file:
        input_data = ''.join(input_data_file.readlines())
    #print input_data
    graph = parse_input(input_data)
    n_nodes  = len(graph)
    
    ## Traditional way
    n_colors = 4
    colors = dict()
    for node in graph.keys():
        colors[node] = range(n_colors)
#    csp = CSP(graph, colors)
#    result = BFS(csp)
#    if isinstance(result, CSP):
#        print '# colors needed = %d' % n_colors
#        print create_output(n_colors, result)
#        print 'check constraints: ', check_constraints(result)
#    else: 
#        print 'No solution with %d colors.' % n_colors
    
    # Greedy search    
    csp = CSP(graph, colors)
    config = Config(graph)
#    n_colors, config = greedy(config)
#    print create_output(n_colors, result)
#    print '# violations: ', n_violations(result, node=0)    
    
    # Local Search
    #result = hill_climbing(result, max_colors=4, max_iter=100, initial='random')
    result = tabu_search2(config, max_colors=4, max_iter=10000, max_lifetime=1024)
    print create_output(result.n_colors(), result)
    print '# violations: ', total_violations(result)
