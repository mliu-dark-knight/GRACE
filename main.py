import subprocess
from config import args
from predictor import Predictor


if __name__ == '__main__':
	subprocess.call('rm ' + args.model_dir + '*', shell=True)
	predictor = Predictor(args)
	predictor.train()
	# predictor.plot()
	predictor.evaluate()
	# predictor.dump()
