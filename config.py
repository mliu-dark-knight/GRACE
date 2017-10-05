import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--feat_dim', default=256, help='Feature dimension')
	parser.add_argument('--embed_dim', default=64, help='Embedding dimension')
	parser.add_argument('--encoder_hidden', default=[128], help='Encoder hidden layer dimension')
	parser.add_argument('--decoder_hidden', default=[128], help='Decoder hidden layer dimension')
	parser.add_argument('--lambda_r', default=1.0, help='Reconstruct loss coefficient')
	parser.add_argument('--lambda_c', default=1.0, help='Clustering loss coefficient')
	parser.add_argument('--learning_rate', default=1e-3, help=None)
	parser.add_argument('--epoch', default=2, help=None)
	parser.add_argument('--step', default=200, help=None)
	parser.add_argument('--dataset', default='cora', help=None)
	return parser.parse_args()

args = parse_args()
args.feature_file = 'data/' + args.dataset + '/feature.txt'
args.edge_file = 'data/' + args.dataset + '/edge.txt'
args.cluster_file = 'data/' + args.dataset + '/cluster.txt'
