import numpy as np
import argparse
from subprocess import *
from tensorflow.python.client import device_lib


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_exp', type=int, default=10, help='Number of experiment')
	return parser.parse_args()


def run(num_exp, arg=None, val=None):
	num_device = sum(1 for device in device_lib.list_local_devices() if device.device_type == 'GPU')
	num_device = 1
	f1, jc, nmi = [], [], []
	processes = []
	batch_processes = []
	for i in range(num_exp):
		device_id = -1 if num_device == 0 else i % num_device
		if arg:
			process = Popen('python2 main.py --%s %s --device %d' % (arg, val, device_id), shell=True, stdout=PIPE)
		else:
			process = Popen('python2 main.py --device %d' % (device_id), shell=True, stdout=PIPE)
		processes.append(process)
		if num_device != 0:
			batch_processes.append(process)
		if num_device != 0 and len(batch_processes) == num_device:
			for j in range(20):
				print
			for process in batch_processes:
				process.wait()
			for process in batch_processes:
				assert process.poll() is None
			batch_processes = []
			for j in range(20):
				print
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
