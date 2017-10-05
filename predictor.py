import tensorflow as tf
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from utils import Graph, plot
from DEC import DEC


class Predictor(object):
	def __init__(self, paras):
		self.paras = paras
		self.graph = Graph.load_graph(paras.feature_file, paras.edge_file, paras.label_file)

	def train(self):
		model = DEC(self.paras, self.graph)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			Z = sess.run(model.encode)
			kmeans = KMeans(n_clusters=self.paras.num_cluster).fit(Z)
			model.init_mean(kmeans.cluster_centers_)
			for _ in tqdm(range(self.paras.epoch), ncols=100):
				P = model.get_P(sess)
				for _ in range(self.paras.step):
					sess.run(model.gradient_descent, feed_dict=P)
		embedding = model.get_embedding(sess)
		self.dump(embedding)
		plot(self.TSNE(embedding), self.graph.label)

	def dump(self, embedding):
		pass

	def TSNE(self, embedding):
		return TSNE(n_components=2).fit_transform(embedding)
