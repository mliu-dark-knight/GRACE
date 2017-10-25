import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv


class Graph(object):
	def __init__(self, feature_file, edge_file, cluster_file, alpha, lambda_):
		self.init(feature_file, edge_file, cluster_file, alpha, lambda_)

	def init(self, feature_file, edge_file, cluster_file, alpha, lambda_):
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

		edges = defaultdict(set)
		with open(edge_file) as f:
			for line in f:
				tuple = list(map(int, line.rstrip().split(',')))
				assert len(tuple) == 2
				edges[tuple[0]].add(tuple[1])
				edges[tuple[1]].add(tuple[0])
		for v, ns in edges.items():
			if v not in ns:
				edges[v].add(v)

		indices = []
		T_values, RI_values, RW_values = [], [], []

		for v, ns in edges.items():
			indices.append(np.array([v, v]))
			T_values.append(1.0 / len(ns))
			RI_values.append(1.0 - alpha / len(ns))
			RW_values.append(1.0 - (1.0 - lambda_) / len(ns))
			for n in ns:
				if v != n:
					indices.append(np.array([v, n]))
					T_values.append(1.0 / len(ns))
					RI_values.append(-alpha / len(ns))
					RW_values.append(-(1.0 - lambda_) / len(ns))

		self.indices = np.array(indices)
		self.T_values = np.asarray(T_values, dtype=np.float32)

		self.RI = inv(csc_matrix((RI_values, (self.indices[:, 1], self.indices[:, 0])), shape=(len(edges), len(edges)))).todense()
		self.RW = lambda_ * inv(csc_matrix((RW_values, (self.indices[:, 0], self.indices[:, 1])), shape=(len(edges), len(edges)))).todense()
		self.RW /= np.sum(self.RW, axis=0)


def load_graph(feature_file, graph_file, cluster_file, alpha, lambda_):
	return Graph(feature_file, graph_file, cluster_file, alpha, lambda_)


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
