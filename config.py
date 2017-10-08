import argparse
import getpass
import sys

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--feat_dim', default=-1, help='Feature dimension')
	parser.add_argument('--embed_dim', default=256, help='Embedding dimension')
	parser.add_argument('--encoder_hidden', default=[1024], help='Encoder hidden layer dimension')
	parser.add_argument('--decoder_hidden', default=[1024], help='Decoder hidden layer dimension')
	parser.add_argument('--random_walk_step', default=2, help=None)
	parser.add_argument('--stay_prob', default=0.8, help='Probability of staying at current node')
	parser.add_argument('--lambda_r', default=0.1, help='Reconstruct loss coefficient')
	parser.add_argument('--lambda_c', default=1.0, help='Clustering loss coefficient')
	parser.add_argument('--learning_rate', default=1e-3, help=None)
	parser.add_argument('--pre_epoch', default=20, help=None)
	parser.add_argument('--pre_step', default=1, help=None)
	parser.add_argument('--epoch', default=20, help=None)
	parser.add_argument('--step', default=1, help=None)
	parser.add_argument('--dataset', default='cora', help=None)
	return parser.parse_args()

args = parse_args()
data_dir = 'data/' + args.dataset + '/' if sys.platform == 'darwin' else \
	'/shared/data/' + getpass.getuser() + '/DEC/' + args.dataset + '/'
args.feature_file = data_dir + 'feature.txt'
args.edge_file = data_dir + 'edge.txt'
args.cluster_file = data_dir + '/cluster.txt'
args.model_file = data_dir + '/model.pkl'
args.plot_file = data_dir + '/plot.png'
args.predict_file = data_dir + '/prediction.txt'
