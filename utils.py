import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


class Graph(object):
	def __init__(self, feature_file, edge_file, cluster_file, stay_prob):
		self.indices, self.values = None, None
		self.feature, self.cluster = None, None
		self.init(feature_file, edge_file, cluster_file, stay_prob)

	def init(self, feature_file, edge_file, cluster_file, stay_prob):
		self.feature = []
		with open(feature_file) as f:
			for line in f:
				self.feature.append(np.array(list(map(int, line.rstrip().split(',')))))
		self.feature = np.array(self.feature)

		self.cluster = []
		with open(cluster_file) as f:
			for line in f:
				self.cluster.append(np.array(list(map(int, line.rstrip().split(',')))))
		self.cluster = np.array(self.cluster)

		edges = defaultdict(list)
		with open(edge_file) as f:
			for line in f:
				tuple = list(map(int, line.rstrip().split(',')))
				assert len(tuple) == 2
				edges[tuple[0]].append(tuple[1])
				edges[tuple[1]].append(tuple[0])

		self.indices, self.values = [], []
		for v, ns in edges.items():
			self.indices.append(np.array([v, v]))
			self.values.append(stay_prob)
			for n in ns:
				self.indices.append(np.array([v, n]))
				self.values.append((1.0 - stay_prob) / len(ns))
		self.indices, self.values = np.array(self.indices), np.asarray(self.values, dtype=np.float32)

	@staticmethod
	def load_graph(feature_file, graph_file, cluster_file, stay_prob):
		return Graph(feature_file, graph_file, cluster_file, stay_prob)


def scatter(data, cluster, plot_file):
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	assert len(set(cluster)) <= len(colors)
	points_set = []
	for cl in range(len(set(cluster))):
		points_set.append(np.array([p for p, c in zip(data, cluster) if c == cl]))
	plt.figure()
	for i, points in enumerate(points_set):
		plt.scatter(points[:, 0], points[:, 1], c=colors[i])
	plt.savefig(plot_file)
	plt.close()

def plot(data, plot_file):
	plt.figure()
	plt.plot(range(len(data)), data)
	plt.savefig(plot_file)
	plt.close()
