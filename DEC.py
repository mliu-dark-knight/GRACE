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
		dense_shape = [self.paras.num_node, self.paras.num_node]
		# random walk outgoing
		self.T1 = tf.SparseTensor(indices=graph.T1_indices, values=graph.T1_values, dense_shape=dense_shape)
		# random walk incoming
		self.T2 = tf.SparseTensor(indices=graph.T2_indices, values=graph.T2_values, dense_shape=dense_shape)
		# graph Laplacian 1
		self.L1 = tf.SparseTensor(indices=graph.L1_indices, values=graph.L1_values, dense_shape=dense_shape)
		# graph Laplacian 2
		self.L2 = tf.SparseTensor(indices=graph.L2_indices, values=graph.L2_values, dense_shape=dense_shape)
		# influence propagation outgoing matrix
		self.RI1 = tf.Variable(graph.RI1, trainable=False, dtype=tf.float32)
		# influence propagation incoming matrix
		self.RI2 = tf.Variable(graph.RI2, trainable=False, dtype=tf.float32)
		self.mean = weight('mean', [self.paras.num_cluster, self.paras.embed_dim])
		self.P = tf.placeholder(tf.float32, [self.paras.num_node, self.paras.num_cluster])
		self.Z = self.encode()
		self.Z_transform = self.transform()
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
		Z = self.Z_transform
		Z = tf.tile(tf.expand_dims(Z, 1), tf.stack([1, self.paras.num_cluster, 1]))
		Q = 1.0 / (tf.reduce_sum(tf.squared_difference(Z, self.mean), axis=2) + 1.0)
		return Q / tf.reduce_sum(Q, axis=1, keep_dims=True)

	def loss_r(self, X_p):
		# todo: check this
		return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=X_p), axis=1))

	def loss_c(self):
		loss_c = tf.reduce_mean(self.P * tf.log(self.P / self.Q))
		loss_c = tf.verify_tensor_all_finite(loss_c, 'check nan')
		return loss_c

	def transform(self):
		transition_function = self.paras.transition_function
		if transition_function in ['T1', 'T2']:
			for i in range(self.paras.random_walk_step):
				Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), self.Z)
		elif transition_function in ['L1', 'L2']:
			Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), self.Z)
		elif transition_function in ['RI1', 'RI2']:
			Z = tf.matmul(self.__getattribute__(transition_function), self.Z, transpose_a=True)
		else:
			raise ValueError('Invalid transition function')
		return Z

	def encode(self):
		hidden = self.X
		for i, dim in enumerate(self.paras.encoder_hidden + [self.paras.embed_dim]):
			hidden = fully_connected(hidden, dim, 'encoder_' + str(i))
		return hidden

	def decode(self):
		hidden = self.transform()
		for i, dim in enumerate(self.paras.decoder_hidden):
			hidden = fully_connected(hidden, dim, 'decoder_' + str(i))
		return fully_connected(hidden, self.paras.feat_dim, 'decoder_' + str(len(self.paras.decoder_hidden)), activation='linear')

	def get_embedding(self, sess):
		return sess.run(self.Z)

	def init_mean(self, mean, sess):
		sess.run(self.mean.assign(mean))

	def get_P(self, sess):
		P = tf.square(self.Q) / tf.reduce_sum(self.Q, axis=0)
		return sess.run(P / tf.reduce_sum(P, axis=1, keep_dims=True))

	def predict(self, sess):
		return sess.run(tf.one_hot(tf.argmax(self.Q, axis=1), depth=self.paras.num_cluster, on_value=1, off_value=0))
