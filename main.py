from __future__ import print_function
import os
import subprocess
import numpy as np
from config import *
from predictor import *


if __name__ == '__main__':
	predictors = initialize_predictors(args)
	f1_list, jc_list, nmi_list = [], [], []
	for predictor in predictors:
		predictor.train()
		# predictor.plot()
		f1, jc, nmi = predictor.evaluate()
		f1_list.append(f1)
		jc_list.append(jc)
		nmi_list.append(nmi)
		# predictor.dump()
	print('f1 score %f' % np.mean(f1_list))
	print('jc score %f' % np.mean(jc_list))
	print('nmi score %f' % np.mean(nmi_list))
