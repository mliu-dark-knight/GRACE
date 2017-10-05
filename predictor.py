import pickle
import tensorflow as tf
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from utils import *
from DEC import DEC


class Predictor(object):
	def __init__(self, paras):
		self.paras = deepcopy(paras)
		self.graph = Graph.load_graph(paras.feature_file, paras.edge_file, paras.cluster_file, paras.stay_prob)
		self.reset_paras()

	def reset_paras(self):
		self.paras.feat_dim = len(self.graph.feature[0])
		self.paras.num_node = len(self.graph.feature)
		self.paras.num_cluster = len(self.graph.cluster[0])

	def train(self):
		model = DEC(self.paras, self.graph)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			Z = sess.run(model.Z)
			kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(Z)
			model.init_mean(kmeans.cluster_centers_, sess)
			for _ in tqdm(range(self.paras.epoch), ncols=100):
				P = model.get_P(sess)
				for _ in range(self.paras.step):
					sess.run(model.gradient_descent, feed_dict={model.P: P})
			self.embedding = model.get_embedding(sess)

	def plot(self):
		plot(self.tSNE(), np.argmax(self.graph.cluster, axis=1), self.paras.plot_file)

	def evaluate(self):
		kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(self.embedding)
		self.prediction = OneHotEncoder().fit_transform(kmeans.labels_)

	def dump(self):
		pickle.dump(self.embedding, open(self.paras.model_file, 'wb'))
		with open(self.paras.predict_file, 'w') as f:
			for prediction in np.transpose(self.prediction):
				f.write(','.join(map(str, prediction)) + '\n')

	def tSNE(self):
		return TSNE(n_components=2).fit_transform(self.embedding)
