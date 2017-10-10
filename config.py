import argparse
import getpass
import sys

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--feat_dim', default=-1, help='Feature dimension')
	parser.add_argument('--embed_dim', default=1024, help='Embedding dimension')
	parser.add_argument('--encoder_hidden', default=[], help='Encoder hidden layer dimension')
	parser.add_argument('--decoder_hidden', default=[], help='Decoder hidden layer dimension')
	parser.add_argument('--transition_function', default='RI1', help='Transition function [T1, T2, L1, L2, RI1, RI2]')
	parser.add_argument('--random_walk_step', default=5, help=None)
	parser.add_argument('--alpha', default=0.1, help='Damping coefficient for propagation process')
	parser.add_argument('--lambda_r', default=1.0, help='Reconstruct loss coefficient')
	parser.add_argument('--lambda_c', default=0.2, help='Clustering loss coefficient')
	parser.add_argument('--learning_rate', default=1e-3, help=None)
	parser.add_argument('--pre_epoch', default=20, help=None)
	parser.add_argument('--pre_step', default=20, help=None)
	parser.add_argument('--epoch', default=10, help=None)
	parser.add_argument('--step', default=10, help=None)
	parser.add_argument('--dataset', default='cora', help=None)
	return parser.parse_args()

args = parse_args()
data_dir = 'data/' + args.dataset + '/' if sys.platform == 'darwin' else \
	'/shared/data/' + getpass.getuser() + '/DEC/' + args.dataset + '/'
args.model_dir = data_dir + 'model/'
args.feature_file = data_dir + 'feature.txt'
args.edge_file = data_dir + 'edge.txt'
args.cluster_file = data_dir + '/cluster.txt'
args.model_file = data_dir + '/model.pkl'
args.plot_file = data_dir + '/plot.png'
args.predict_file = data_dir + '/prediction.txt'
