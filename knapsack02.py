#!/usr/bin python
# Sun Jun 26 10:18:27 2016

"""
26 Jun 2016

This is a second attempt to solve the knapsack problem, after the efforts
in November 2015.
This file only implements Branch&Bound, recognizing that Dynamic Programming
is too demanding on memory.

Still to implemented:
    * use heapq for faster sorting with best-first search: OK
    * implement limited discrepancy search (LDS)
"""

from __future__ import division

import time, copy, sys
import heapq # for optimal queueing
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

import pdb

class G: # Globals
    node_count = 0

class Node:
    count = 0
    def __init__(self, parent=None, choice=0, items=None):
        #pdb.set_trace()
        self.count = Node.count
        Node.count += 1
        if parent == None:
            self.level = 0
            self.room  = 0
            self.value = 0
            self.upper_bound = 0
            self.items = []
        else:
            self.items = parent.items[:] 
            self.items.append(choice)
            self.level = parent.level + 1
            self.room  = parent.room - choice*items[self.level-1].weight
            self.value = parent.value + choice*items[self.level-1].value
        
def timeit(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        print '%s -> %.2fs' % (fn.__name__, time.time()-start)
        return result
    return wrapper

def read_input(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    return item_count, capacity, items
 
def straight_forward_UB(node, items):
    UB = node.value # start out with current value
    for i in range(node.level, len(items)):
        UB += items[i].value
    return UB

def relaxation_UB(node, items):
    UB = node.value     # start out with current value
    room = node.room    # bookkeeping: how much room left?
    for i in range(node.level, len(items)):
        if room >= items[i].weight:
            UB += items[i].value
            room -= items[i].weight
        elif 0 < room < items[i].weight: # still room left, but not enough for item
            UB += items[i].value * node.room / items[i].weight
            room = 0
    return UB
        
@timeit
def search(capacity, items, type='DFS', compute_upper_bound=straight_forward_UB):
    n_items = len(items)
    incumbent_value = 0
    ## Initialise init node
    init_node = Node(parent=None)
    init_node.room  = capacity
    init_node.upper_bound = sum(item.value for item in items)
    nodes = []
    if type=='DFS': # depth-first: push on heap, according to count
        heapq.heappush(nodes, (-init_node.count, init_node))
    elif type=='BEST': # best-first: push on heap, according to upper_bound
        heapq.heappush(nodes, (init_node.upper_bound, init_node))
    leaf_nodes = []
    #pdb.set_trace()
    while nodes:
        _, node = heapq.heappop(nodes)      
        if node.level == n_items: # if the next node is a leaf node
            print 'Optimal: %d [> %d]' % (node.value, incumbent_value)
            incumbent_value = node.value
            leaf_nodes.append(node)
        else: # else, develop the node and discard if need be
            for choice in [0, 1]:
                new_node = Node(node, choice, items)
                if new_node.room >= 0:
                    new_node.upper_bound = compute_upper_bound(new_node, items)
                    if new_node.upper_bound > incumbent_value:  # only accept if potential solution
                                                                # is better than current one
                        if type=='DFS':
                            heapq.heappush(nodes, (-new_node.count, new_node))
                        elif type=='BEST':
                            heapq.heappush(nodes, (new_node.upper_bound, new_node))
                        G.node_count += 1
    return leaf_nodes, nodes

def parse_result(value, taken):
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_it(input_data, function='DFS'):
    _, capacity, items = read_input(input_data)
    # sort items according to value/weight
    items = sorted(items, key=lambda item: item.value / item.weight, reverse=True)
    result, nodes = search(capacity, items, type=function, 
                 compute_upper_bound=relaxation_UB)
    best_result = result[-1]
    new_items = best_result.items[:]
    for i in range(len(items)):
        new_items[items[i].index] = best_result.items[i]
    return parse_result(best_result.value, new_items)# , nodes 

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
    else:
        file_location = './data/ks_45_0'
        
    with open(file_location, 'r') as input_data_file:
        input_data = ''.join(input_data_file.readlines())
    result = solve_it(input_data, function='BEST')
    print result
