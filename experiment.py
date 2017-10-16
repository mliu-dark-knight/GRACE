import argparse
import subprocess
from multiprocessing import *

import numpy as np

from config import args
from predictor import Predictor


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_exp', type=int, default=1, help='Number of experiment')
	parser.add_argument('--num_device', type=int, default=0, help='Number of GPU, change to 0 if not using CPU')

	parser.add_argument('--embed_dim', type=list, default=[256], help='Embedding dimension')
	parser.add_argument('--encoder_hidden', type=list, default=[[1024, 512]], help='Encoder hidden layer dimension')
	parser.add_argument('--keep_prob', type=list, default=[0.4] * 4, help='Keep probability of dropout')
	parser.add_argument('--BN', type=list, default=[False], help='Apply batch normalization')
	parser.add_argument('--lambda_r', type=list, default=[1.0], help='Reconstruct loss coefficient')
	parser.add_argument('--lambda_c', type=list, default=[0.1], help='Clustering loss coefficient')
	parser.add_argument('--optimizer', type=list, default=['Adam'], help='Optimizer [Adam, Momentum, GradientDescent, RMSProp, Adagrad]')
	parser.add_argument('--pre_epoch', type=list, default=[60], help=None)
	parser.add_argument('--pre_step', type=list, default=[60], help=None)
	parser.add_argument('--epoch', type=list, default=[40], help=None)
	parser.add_argument('--step', type=list, default=[40], help=None)
	return parser.parse_args()


def run(num_exp):
	def worker():
		predictor = Predictor(args)
		predictor.train()
		f1, jc, nmi = predictor.evaluate()
		queue.put((f1, jc, nmi))

	subprocess.call('rm ' + args.model_dir + '*', shell=True)
	f1_list, jc_list, nmi_list = [], [], []
	queue = Queue()
	processes = []
	batch_processes = []
	for i in range(num_exp):
		device_id = -1 if local_args.num_device == 0 else i % local_args.num_device
		args.device = device_id
		process = Process(target=worker)
		process.start()
		processes.append(process)
		if local_args.num_device != 0:
			batch_processes.append(process)
		if local_args.num_device != 0 and len(batch_processes) == local_args.num_device:
			for process in batch_processes:
				process.join()
			batch_processes = []
	for process in processes:
		process.join()

	for _ in processes:
		f1, jc, nmi = queue.get()
		f1_list.append(f1)
		jc_list.append(jc)
		nmi_list.append(nmi)

	return np.mean(f1_list), np.std(f1_list), np.mean(jc_list), np.std(jc_list), np.mean(nmi_list), np.std(nmi_list)

if __name__ == '__main__':
	local_args = parse_args()
	for embed_dim in local_args.embed_dim:
		args.embed_dim = embed_dim
		for encoder_hidden in local_args.encoder_hidden:
			args.encoder_hidden, args.decoder_hidden = encoder_hidden, list(reversed(encoder_hidden))
			for keep_prob in local_args.keep_prob:
				args.keep_prob = keep_prob
				for BN in local_args.BN:
					args.BN = BN
					for lambda_r in local_args.lambda_r:
						args.lambda_r = lambda_r
						for lambda_c in local_args.lambda_c:
							args.lambda_c = lambda_c
							for optimizer in local_args.optimizer:
								args.optimizer = optimizer
								for pre_epoch in local_args.pre_epoch:
									args.pre_epoch = pre_epoch
									for pre_step in local_args.pre_step:
										args.pre_step = pre_step
										for epoch in local_args.epoch:
											args.epoch = epoch
											for step in local_args.step:
												args.step = step

												print args
												f1_mean, f1_std, jc_mean, jc_std, nmi_mean, nmi_std = run(local_args.num_exp)
												print 'f1 mean %f, std %f' % (f1_mean, f1_std)
												print 'jc mean %f, std %f' % (jc_mean, jc_std)
												print 'nmi mean %f, std %f' % (nmi_mean, nmi_std)
