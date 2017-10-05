from config import args
from predictor import Predictor


if __name__ == '__main__':
	predictor = Predictor(args)
	predictor.train()
	predictor.evaluate()
	# predictor.plot()
