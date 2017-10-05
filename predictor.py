import pickle
import tensorflow as tf
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from utils import Graph
from DEC import DEC
from evaluate import f1_community, jc_community, nmi_community


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
			embedding = model.get_embedding(sess)
			self.dump(embedding)
			return model.predict(sess)

	def evaluate(self):
		prediction = self.train()
		print 'f1 score %f' % f1_community(prediction, self.graph.cluster)
		print 'jc score %f' % jc_community(prediction, self.graph.cluster)
		print 'nmi score %f' % nmi_community(prediction, self.graph.cluster)

	def dump(self, embedding):
		pickle.dump(embedding, open(self.paras.model_file, 'wb'))

	def TSNE(self, embedding):
		return TSNE(n_components=2).fit_transform(embedding)
