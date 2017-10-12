import numpy as np
import argparse
from subprocess import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_exp', type=int, default=3, help='Number of experiment')
	return parser.parse_args()


def run(num_exp, arg=None, val=None):
	f1, jc, nmi = [], [], []
	processes = []
	for i in range(num_exp):
		if arg:
			process = Popen('python2 main.py --%s %s' % (arg, val))
		else:
			process = Popen('python2 main.py', shell=True, stdout=PIPE)
		processes.append(process)
	for process in processes:
		process.wait()
	for process in processes:
		for line in process.stdout:
			if 'score' in line:
				split = line.rstrip().split()
				type, score = split[0], float(split[2])
				eval(type).append(score)
	return np.mean(f1), np.std(f1), np.mean(jc), np.std(jc), np.mean(nmi), np.std(nmi)

if __name__ == '__main__':
	args = parse_args()
	f1_mean, f1_std, jc_mean, jc_std, nmi_mean, nmi_std = run(args.num_exp)
	print 'f1 mean %f, std %f' % (f1_mean, f1_std)
	print 'jc mean %f, std %f' % (jc_mean, jc_std)
	print 'nmi mean %f, std %f' % (nmi_mean, nmi_std)
