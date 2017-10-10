import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import pickle

class Graph(object):
	def __init__(self, feature_file, edge_file, cluster_file, alpha):
		self.init(feature_file, edge_file, cluster_file, alpha)

	def init(self, feature_file, edge_file, cluster_file, alpha):
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

		indices = []
		T1_values, T2_values, L1_values, L2_values, RI1_values, RI2_values = [], [], [], [], [], []

		for v, ns in edges.items():
			indices.append(np.array([v, v]))
			T1_values.append(1.0 / len(ns))
			T2_values.append(1.0 / len(ns))
			L1_values.append(1.0 - 1.0 / len(ns))
			L2_values.append(1.0 + 1.0 / len(ns))
			RI1_values.append(1.0 - alpha / len(ns))
			RI2_values.append(1.0 - alpha / len(ns))
			for n in ns:
				if v != n:
					indices.append(np.array([v, n]))
					T1_values.append(1.0 / len(ns))
					T2_values.append(1.0 / len(edges[n]))
					L1_values.append(-1.0 / (np.sqrt(len(ns))*np.sqrt(len(edges[n]))))
					L2_values.append(1.0 / (np.sqrt(len(ns))*np.sqrt(len(edges[n]))))
					RI1_values.append(-alpha / len(ns))
					RI2_values.append(-alpha / len(edges[n]))

		self.indices = np.array(indices)
		self.T1_values = np.asarray(T1_values, dtype=np.float32)
		self.T2_values = np.asarray(T2_values, dtype=np.float32)
		self.L1_values = np.asarray(L1_values, dtype=np.float32)
		self.L2_values = np.asarray(L2_values, dtype=np.float32)

		self.RI1 = inv(csc_matrix((RI1_values, (self.indices[:,0],self.indices[:,1])), shape=(len(edges), len(edges)))).todense()
		#self.RI1 = np.genfromtxt('r5.csv', delimiter=',')
		#self.RI1 = np.transpose(self.RI1)
		#self.T1 = csc_matrix((self.T1_values, (self.T1_indices[:,0],self.T1_indices[:,1])), shape=(len(edges), len(edges))).todense()
		#print(np.sum(self.RI1, axis=0))
		#print(np.sum(self.RI1, axis=1))
		self.RI2 = inv(csc_matrix((RI2_values, (self.indices[:,0], self.indices[:,1])), shape=(len(edges), len(edges)))).todense()
		#with open('/home/jiyang3/RI.pkl', 'wb') as f:
		#	pickle.dump(self.RI1, f)
	@staticmethod
	def load_graph(feature_file, graph_file, cluster_file, alpha):
		return Graph(feature_file, graph_file, cluster_file, alpha)


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
