import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import inv


class Graph(object):
	def __init__(self, feature_file, edge_file, cluster_file, stay_prob, alpha, beta):
		self.init(feature_file, edge_file, cluster_file, stay_prob, alpha, beta)

	def init(self, feature_file, edge_file, cluster_file, stay_prob, alpha, beta):
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
			assert v not in ns
			edges[v].add(v)

		T1_indices, T1_values = [], []
		T2_indices, T2_values = [], []
		L1_indices, L1_values = [], []
		L2_indices, L2_values = [], []
		row, col, val = [], [], []
		gamma = alpha / beta
		for v, ns in edges.items():
			T1_indices.append(np.array([v, v]))
			T2_indices.append(np.array([v, v]))
			L1_indices.append(np.array([v, v]))
			L2_indices.append(np.array([v, v]))
			row.append(v)
			col.append(v)
			val.append(1.0 - gamma / len(ns))
			T1_values.append(stay_prob + 1.0 / len(ns))
			T2_values.append(stay_prob + 1.0 / len(ns))
			L1_values.append(1.0)
			L2_values.append(1.0)
			for n in ns:
				if v != n:
					T1_indices.append(np.array([v, n]))
					T2_indices.append(np.array([n, v]))
					L1_indices.append(np.array([v, n]))
					L2_indices.append(np.array([v, n]))
					row.append(v)
					col.append(n)
					val.append(-gamma / len(ns))
					T1_values.append((1.0 - stay_prob) / len(ns))
					T2_values.append((1.0 - stay_prob) / len(ns))
					L1_values.append(1.0 - 1.0 / len(ns))
					L2_values.append(1.0 + 1.0 / len(ns))

		self.T1_indices = np.array(T1_indices)
		self.T2_indices = np.array(T2_indices)
		self.L1_indices = np.array(L1_indices)
		self.L2_indices = np.array(L2_indices)
		self.T1_values = np.asarray(T1_values, dtype=np.float32)
		self.T2_values = np.asarray(T2_values, dtype=np.float32)
		self.L1_values = np.asarray(L1_values, dtype=np.float32)
		self.L2_values = np.asarray(L2_values, dtype=np.float32)

		inverse = csc_matrix((val, (row, col)), shape=(len(edges), len(edges)))
		self.RI = inv(inverse).todense()

	@staticmethod
	def load_graph(feature_file, graph_file, cluster_file, stay_prob, alpha, beta):
		return Graph(feature_file, graph_file, cluster_file, stay_prob, alpha, beta)


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
