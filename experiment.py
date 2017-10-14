import argparse
import subprocess
import numpy as np
from config import args
from predictor import Predictor


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_exp', type=int, default=10, help='Number of experiment')
	return parser.parse_args()


def run(num_exp):
	subprocess.call('rm ' + args.model_dir + '*', shell=True)
	predictor = Predictor(args)

	f1_list, jc_list, nmi_list = [], [], []
	for i in range(num_exp):
		predictor.train()
		f1, jc, nmi = predictor.evaluate()
		f1_list.append(f1)
		jc_list.append(jc)
		nmi_list.append(nmi)

	return np.mean(f1_list), np.std(f1_list), np.mean(jc_list), np.std(jc_list), np.mean(nmi_list), np.std(nmi_list)

if __name__ == '__main__':
	local_args = parse_args()
	f1_mean, f1_std, jc_mean, jc_std, nmi_mean, nmi_std = run(local_args.num_exp)
	print 'f1 mean %f, std %f' % (f1_mean, f1_std)
	print 'jc mean %f, std %f' % (jc_mean, jc_std)
	print 'nmi mean %f, std %f' % (nmi_mean, nmi_std)
