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


class Predictor(object):
	def __init__(self, paras):
		self.paras = deepcopy(paras)
		self.graph = Graph.load_graph(paras.feature_file, paras.edge_file, paras.cluster_file, paras.stay_prob, paras.alpha, paras.beta)
		self.reset_paras()

	def reset_paras(self):
		self.paras.feat_dim = len(self.graph.feature[0])
		self.paras.num_node = len(self.graph.feature)
		self.paras.num_cluster = len(self.graph.cluster[0])

	def train(self):
		model = DEC(self.paras, self.graph)
		with tf.Session() as sess:
			tf.summary.FileWriter(self.paras.model_dir, graph=sess.graph)
			sess.run(tf.global_variables_initializer())
			for _ in tqdm(range(self.paras.pre_epoch), ncols=100):
				for _ in range(self.paras.pre_step):
					sess.run(model.pre_gradient_descent)
			print('reconstruction loss: %f' % sess.run(model.loss_r))

			Z = sess.run(model.Z)
			kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(Z)
			model.init_mean(kmeans.cluster_centers_, sess)

			self.diff = []
			s_prev = model.predict(sess)
			for _ in tqdm(range(self.paras.epoch), ncols=100):
				P = model.get_P(sess)
				for _ in range(self.paras.step):
					sess.run(model.gradient_descent, feed_dict={model.P: P})
				s = model.predict(sess)
				self.diff.append(np.sum(s_prev != s) / 2.0)
				s_prev = s
			P = model.get_P(sess)
			print('clustering loss: %f' % sess.run(model.loss_c, feed_dict={model.P: P}))
			self.embedding = model.get_embedding(sess)
			self.prediction = model.predict(sess)
			# kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(self.embedding)
			# self.prediction = MultiLabelBinarizer().fit_transform([[label] for label in kmeans.labels_])

	def plot(self):
		# scatter(self.tSNE(), np.argmax(self.graph.cluster, axis=1), self.paras.plot_file)
		plot(self.diff, self.paras.plot_file)

	def evaluate(self):
		prediction, ground_truth = np.transpose(self.prediction), np.transpose(self.graph.cluster)
		print('f1 score %f' % f1_community(prediction, ground_truth))
		print('jc score %f' % jc_community(prediction, ground_truth))
		print('nmi score %f' % nmi_community(prediction, ground_truth))

	def dump(self):
		# pickle.dump(self.embedding, open(self.paras.model_file, 'wb'))
		with open(self.paras.predict_file, 'w') as f:
			for prediction in self.prediction:
				f.write(','.join(map(str, prediction)) + '\n')

	def tSNE(self):
		return TSNE(n_components=2).fit_transform(self.embedding)
