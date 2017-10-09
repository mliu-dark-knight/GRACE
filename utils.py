import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


class Graph(object):
	def __init__(self, feature_file, edge_file, cluster_file, stay_prob):
		self.init(feature_file, edge_file, cluster_file, stay_prob)

	def init(self, feature_file, edge_file, cluster_file, stay_prob):
		feature = []
		with open(feature_file) as f:
			for line in f:
				feature.append(np.array(list(map(int, line.rstrip().split(',')))))
		self.feature = np.array(feature)

		cluster = []
		with open(cluster_file) as f:
			for line in f:
				cluster.append(np.array(list(map(int, line.rstrip().split(',')))))
		self.cluster = np.array(cluster)

		edges = defaultdict(list)
		with open(edge_file) as f:
			for line in f:
				tuple = list(map(int, line.rstrip().split(',')))
				assert len(tuple) == 2
				edges[tuple[0]].append(tuple[1])
				edges[tuple[1]].append(tuple[0])

		indices, T_values, L1_values, L2_values = [], [], [], []
		for v, ns in edges.items():
			indices.append(np.array([v, v]))
			T_values.append(stay_prob)
			L1_values.append(1.0)
			L2_values.append(1.0)
			for n in ns:
				indices.append(np.array([v, n]))
				T_values.append((1.0 - stay_prob) / len(ns))
				L1_values.append(1.0 - 1.0 / len(ns))
				L2_values.append(1.0 + 1.0 / len(ns))
		self.indices = np.array(indices)
		self.T_values = np.asarray(T_values, dtype=np.float32)
		self.L1_values = np.asarray(L1_values, dtype=np.float32)
		self.L1_values = np.asarray(L1_values, dtype=np.float32)
		self.L2_values = np.asarray(L2_values, dtype=np.float32)

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
