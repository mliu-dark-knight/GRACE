import subprocess
from config import args
from predictor import Predictor


if __name__ == '__main__':
	subprocess.call('rm ' + args.model_dir + '*', shell=True)
	predictor = Predictor(args)
	predictor.train()
	# predictor.plot()
	f1, jc, nmi = predictor.evaluate()
	print 'f1 score %f' % f1
	print 'jc score %f' % jc
	print 'nmi score %f' % nmi
	# predictor.dump()
