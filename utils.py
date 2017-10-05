import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


class Graph(object):
	def __init__(self, feature_file, edge_file, label_file):
		self.indices, self.values = None, None
		self.feature, self.label = None, None
		self.init(feature_file, edge_file, label_file)

	def init(self, feature_file, edge_file, label_file):
		self.feature = []
		with open(feature_file) as f:
			for line in f:
				self.feature.append(map(int, line.rstrip().split()))
		self.label = []
		with open(label_file) as f:
			for line in f:
				self.label.append(int(line.rstrip()))

		edges = defaultdict(list)
		with open(edge_file) as f:
			for line in f:
				tuple = map(int, line.rstrip().split())
				assert len(tuple) == 2
				edges[tuple[0]].append(tuple[1])
				edges[tuple[1]].append(tuple[0])

		self.indices, self.values = [], []
		for v, ns in edges.items():
			for n in ns:
				self.indices.append([v, n])
				self.values.append(1.0 / ns)

	@staticmethod
	def load_graph(feature_file, graph_file, label_file):
		return Graph(feature_file, graph_file, label_file)



def plot(embedding, label):
	pass
