from __future__ import print_function
import pickle
import tensorflow as tf
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from utils import *
from DEC import DEC
from evaluate import f1_community, jc_community, nmi_community
from config import args

class CPIP(object):
	def __init__(self, paras):
		self.paras = deepcopy(paras)
		self.graph = Graph.load_graph(paras.feature_file, paras.edge_file, paras.cluster_file, paras.alpha)
		self.reset_paras()

		dense_shape = [self.paras.num_node, self.paras.num_node]
		# random walk outgoing
		self.T1 = tf.SparseTensor(indices=self.graph.indices, values=self.graph.T1_values, dense_shape=dense_shape)
		# random walk incoming
		self.T2 = tf.SparseTensor(indices=self.graph.indices, values=self.graph.T2_values, dense_shape=dense_shape)
		# graph Laplacian 1
		self.L1 = tf.SparseTensor(indices=self.graph.indices, values=self.graph.L1_values, dense_shape=dense_shape)
		# graph Laplacian 2
		self.L2 = tf.SparseTensor(indices=self.graph.indices, values=self.graph.L2_values, dense_shape=dense_shape)
		# influence propagation outgoing matrix
		self.RI1 = tf.Variable(self.graph.RI1, trainable=False, dtype=tf.float32)
		# influence propagation incoming matrix
		self.RI2 = tf.Variable(self.graph.RI2, trainable=False, dtype=tf.float32)

	def reset_paras(self):
		self.paras.feat_dim = len(self.graph.feature[0])
		self.paras.num_node = len(self.graph.feature)
		self.paras.num_cluster = len(self.graph.cluster[0])

	def train(self):
		with tf.Session() as sess:
			tf.summary.FileWriter(self.paras.model_dir, graph=sess.graph)
			X = tf.Variable(self.graph.feature, trainable=False, dtype=tf.float32)
			Z = self.transform(X)
			sess.run(tf.global_variables_initializer())
			Z = sess.run(Z)
			kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(Z)
			self.prediction = MultiLabelBinarizer().fit_transform([[label] for label in kmeans.labels_])

	def evaluate(self):
		prediction, ground_truth = np.transpose(self.prediction), np.transpose(self.graph.cluster)
		print('f1 score %f' % f1_community(prediction, ground_truth))
		print('jc score %f' % jc_community(prediction, ground_truth))
		print('nmi score %f' % nmi_community(prediction, ground_truth))

	def transform(self, X):
		transition_function = self.paras.transition_function
		Z = X
		if transition_function in ['T1', 'T2']:
			for i in range(self.paras.random_walk_step):
				Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), Z)
		elif transition_function in ['L1', 'L2']:
			Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), Z)
		elif transition_function in ['RI1', 'RI2']:
			Z = tf.matmul(self.__getattribute__(transition_function), Z)
		else:
			raise ValueError('Invalid transition function')
		return Z

if __name__ == "__main__":
	cpip = CPIP(args)
	cpip.train()
	cpip.evaluate()