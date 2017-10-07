from NN import *

class DEC(object):
	def __init__(self, paras, graph):
		self.paras = paras
		self.build(graph)

	def build(self, graph):
		self.build_variable(graph)
		self.build_loss()

	def build_variable(self, graph):
		self.X = tf.Variable(graph.feature, trainable=False, dtype=tf.float32)
		self.graph = tf.SparseTensor(indices=graph.indices, values=graph.values, dense_shape=[self.paras.num_node, self.paras.num_node])
		self.mean = weight('mean', [self.paras.num_cluster, self.paras.embed_dim])
		self.P = tf.placeholder(tf.float32, [self.paras.num_node, self.paras.num_cluster])
		self.Z = self.encode()
		self.Q = self.build_Q()

	def build_loss(self):
		X_p = self.decode()
		self.loss_r, self.loss_c = self.loss_r(X_p), self.loss_c()
		pre_loss = self.loss_r
		pre_optimizer = tf.train.AdamOptimizer(learning_rate=self.paras.learning_rate)
		self.pre_gradient_descent = pre_optimizer.minimize(pre_loss)
		loss = self.paras.lambda_r * self.loss_r + self.paras.lambda_c * self.loss_c
		optimizer = tf.train.AdamOptimizer(learning_rate=self.paras.learning_rate)
		self.gradient_descent = optimizer.minimize(loss)

	def build_Q(self):
		Z = self.random_walk()
		Z = tf.tile(tf.expand_dims(Z, 1), tf.stack([1, self.paras.num_cluster, 1]))
		Q = 1.0 / (tf.reduce_sum(tf.squared_difference(Z, self.mean), axis=2) + 1.0)
		return Q / tf.reduce_sum(Q, axis=1, keep_dims=True)

	def loss_r(self, X_p):
		return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.X, X_p), axis=1))

	def loss_c(self):
		loss_c = tf.reduce_sum(self.P * tf.log(self.P / self.Q))
		loss_c = tf.verify_tensor_all_finite(loss_c, 'check nan')
		return loss_c

	def random_walk(self):
		Z = self.Z
		for i in range(self.paras.random_walk_step):
			Z = tf.sparse_tensor_dense_matmul(self.graph, Z)
		return Z

	def encode(self):
		hidden = self.X
		for i, dim in enumerate(self.paras.encoder_hidden + [self.paras.embed_dim]):
			hidden = fully_connected(hidden, dim, 'encoder_' + str(i))
		return hidden

	def decode(self):
		hidden = self.Z
		for i, dim in enumerate(self.paras.decoder_hidden):
			hidden = fully_connected(hidden, dim, 'decoder_' + str(i))
		return fully_connected(hidden, self.paras.feat_dim, 'decoder_' + str(len(self.paras.decoder_hidden)), activation='sigmoid')

	def get_embedding(self, sess):
		return sess.run(self.Z)

	def init_mean(self, mean, sess):
		sess.run(self.mean.assign(mean))

	def get_P(self, sess):
		P = tf.square(self.Q) / tf.reduce_sum(self.Q, axis=0)
		return sess.run(P / tf.reduce_sum(P, axis=1, keep_dims=True))

	def predict(self, sess):
		return sess.run(tf.one_hot(tf.argmax(self.Q, axis=1), depth=self.paras.num_cluster, on_value=1, off_value=0))
