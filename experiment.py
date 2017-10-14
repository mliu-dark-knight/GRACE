import argparse
import sys
import time
import subprocess
import numpy as np
from config import args
from predictor import Predictor
from multiprocessing import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_exp', type=int, default=10, help='Number of experiment')
	parser.add_argument('--num_device', type=int, default=4, help='Number of GPU, change to 0 if not using CPU')
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
		if local_args.num_device != 0:
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

	for process in processes:
		f1, jc, nmi = queue.get()
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
