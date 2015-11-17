#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

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

def output_data(value, taken):
	# prepare the solution in the specified output format
    output = str(value) + ' ' + str(0) + '\n'
    output += ' '.join(map(str, taken))
    return output

def greedy(capacity, items):
	total_value  = 0
	total_weight = 0
	taken = [0]*len(items)
	
	items.sort(key=lambda item: item.value / item.weight)
	
	while items:
		item = items.pop()
		if total_weight + item.weight > capacity:
			continue
		total_weight += item.weight
		total_value  += item.value
		taken[item.index] = 1
	
	return total_value, taken

def dynamic(capacity, items):
	items.sort(key=lambda item: item.index)
	table = [[None]*len(items)]*(capacity+1)
	
	def optimal(k, j):
		if j == 0:
			return 0
		elif table[k][j] is not None:
			return table[k][j]
		else:
			item = items[j]
			if item.weight <= k:
				return max(optimal(k, j-1), item.value + optimal(k-item.weight, j-1))
			else:
				return optimal(k, j-1)
	
	return optimal(capacity, len(items)-1)

def best_first(node):
	return relaxation(items, node)

def branchbound(capacity, items, sort_fn=None):
	class Node:
		def __init__(self, parent, state):
			self.parent = parent
			self.state = state
			self.next = len(state)
			if parent:
				if state[-1] == 1:
					item = find(items, len(state) - 1)
					self.value  = self.parent.value  + item.value
					self.weight = self.parent.weight + item.weight
				else:
					self.value = self.parent.value
					self.weight = self.parent.weight
			else:
				self.value  = 0
				self.weight = 0
	
	def find(items, n):
		items2 = sorted(items, key=lambda item: item.index)
		return items2[n]
	
	def relaxation(items, node): # items are sorted descending according to vi/wi
		value  = node.value
		weight = node.weight
		rest_cap = capacity - weight

		for j in range(len(items)):
			if items[j].index < len(node.state):
				continue
			if items[j].weight <= rest_cap:
				value += items[j].value
				weight += items[j].weight
				rest_cap -= items[j].weight
			elif items[j].weight > rest_cap:
				value += items[j].value * rest_cap/items[j].weight
				break
		return value
	
	def terminal(node):
		return len(node.state) == len(items)
	
	def best_first(node):
		return -relaxation(items, node)
	
	items.sort(key = lambda item: item.value/item.weight)
	initial = Node(parent=None, state=[])
	queue = [initial]
	best_node, best_value = None, 0
	while queue:
		if sort_fn == 'best_first':
			queue.sort(key = best_first)
		node = queue.pop()
		for successor in [0,1]:
			next_node = Node(parent=node, state=node.state + [successor])
			if next_node.weight < capacity:
				if terminal(next_node):
					if next_node.value > best_value:
						best_node = next_node
						best_value = next_node.value
				elif relaxation(items, next_node) > best_value:
					queue.append(next_node)			
	return best_node.value, best_node.state		
			
			
import sys

if __name__ == '__main__':
	file_location = './data/ks_40_0'
	with open(file_location, 'r') as input_data_file:
		input_data = ''.join(input_data_file.readlines())
	_, capacity, items = read_input(input_data)

	value,  taken = greedy(capacity, items[:])
	print value, taken
	
	value, taken = branchbound(capacity, items[:], sort_fn='best_first')
	print value, taken
	
	
