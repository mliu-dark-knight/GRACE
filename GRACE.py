from NN import *

class GRACE(object):
	def __init__(self, paras, graph):
		self.paras = paras
		self.build(graph)

	def build(self, graph):
		self.build_variable(graph)
		self.build_loss()

	def build_variable(self, graph):
		self.training = tf.placeholder(tf.bool)
		self.X = tf.Variable(graph.feature, trainable=False, dtype=tf.float32)
		dense_shape = [self.paras.num_node, self.paras.num_node]
		# random walk outgoing
		self.T1 = tf.SparseTensor(indices=graph.indices, values=graph.T1_values, dense_shape=dense_shape)
		# random walk incoming
		self.T2 = tf.SparseTensor(indices=graph.indices, values=graph.T2_values, dense_shape=dense_shape)
		# graph Laplacian 1
		self.L1 = tf.SparseTensor(indices=graph.indices, values=graph.L1_values, dense_shape=dense_shape)
		# graph Laplacian 2
		self.L2 = tf.SparseTensor(indices=graph.indices, values=graph.L2_values, dense_shape=dense_shape)
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
		self.loss_r, self.loss_c, self.loss_2 = self.build_loss_r(X_p), self.build_loss_c(), self.build_loss_2()
		pre_loss = self.loss_r
		pre_optimizer = getattr(tf.train, self.paras.optimizer + 'Optimizer')(learning_rate=self.paras.learning_rate)
		self.pre_gradient_descent = pre_optimizer.minimize(pre_loss)
		loss = self.paras.lambda_r * self.loss_r + self.paras.lambda_c * self.loss_c + self.paras.lambda_2 * self.loss_2
		optimizer = getattr(tf.train, self.paras.optimizer + 'Optimizer')(learning_rate=self.paras.learning_rate)
		self.gradient_descent = optimizer.minimize(loss)

	def build_Q(self):
		Z = self.Z_transform
		Z = tf.tile(tf.expand_dims(Z, 1), tf.stack([1, self.paras.num_cluster, 1]))
		Q = tf.pow(tf.reduce_sum(tf.squared_difference(Z, self.mean), axis=2) / self.paras.epsilon + 1.0, -(self.paras.epsilon + 1.0) / 2.0)
		return Q / tf.reduce_sum(Q, axis=1, keep_dims=True)

	def build_loss_r(self, X_p):
		# todo: check this
		return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=X_p), axis=1))

	def build_loss_c(self):
		loss_c = tf.reduce_mean(self.P * tf.log(self.P / self.Q))
		loss_c = tf.verify_tensor_all_finite(loss_c, 'check nan')
		return loss_c

	def build_loss_2(self):
		mean = tf.matmul(self.Q, self.Z_transform, transpose_a=True)
		norm = tf.transpose(tf.reduce_sum(self.Q, axis=0, keep_dims=True))
		mean = mean / norm
		return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.mean, mean), axis=1))

	def transform(self):
		transition_function = self.paras.transition_function
		Z = self.Z
		if transition_function in ['T1', 'T2']:
			for i in range(self.paras.random_walk_step):
				Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), self.Z)
		elif transition_function in ['L1', 'L2']:
			Z = tf.sparse_tensor_dense_matmul(self.__getattribute__(transition_function), self.Z)
		elif transition_function in ['RI1', 'RI2']:
			Z = tf.matmul(self.__getattribute__(transition_function), self.Z)
		else:
			raise ValueError('Invalid transition function')
		if self.paras.BN:
			Z = batch_normalization(Z, 'Z')
		return Z

	def encode(self):
		hidden = self.X
		for i, dim in enumerate(self.paras.encoder_hidden + [self.paras.embed_dim]):
			hidden = fully_connected(hidden, dim, 'encoder_' + str(i))
			hidden = dropout(hidden, self.paras.keep_prob, self.training)
		return hidden

	def decode(self):
		hidden = self.Z
		for i, dim in enumerate(self.paras.decoder_hidden):
			hidden = fully_connected(hidden, dim, 'decoder_' + str(i))
			hidden = dropout(hidden, self.paras.keep_prob, self.training)
		return fully_connected(hidden, self.paras.feat_dim, 'decoder_' + str(len(self.paras.decoder_hidden)), activation='linear')

	def get_embedding(self, sess):
		return sess.run(self.Z, feed_dict={self.training: False})

	def init_mean(self, mean, sess):
		sess.run(self.mean.assign(mean))

	def get_P(self, sess):
		P = tf.square(self.Q) / tf.reduce_sum(self.Q, axis=0)
		return sess.run(P / tf.reduce_sum(P, axis=1, keep_dims=True), feed_dict={self.training: False})

	def predict(self, sess):
		return sess.run(tf.one_hot(tf.argmax(self.Q, axis=1), depth=self.paras.num_cluster, on_value=1, off_value=0), feed_dict={self.training: False})
