import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# input = np.array([[0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0],
#        [0, 0, 0, 1, 0],
#        [0, 0, 0, 1, 0],
#        [0, 1, 0, 0, 0]])
# print(np.argwhere(input == 1)[:,-1])
#
# print(np.arange(1,100,1))
# for i in range(100):
# m = scipy.stats.norm(.001,.001)
# distribution = m.rvs(100)[>0]
# distribution = distribution[distribution>0]
# print(distribution)
# x = np.linspace(m.ppf(.01),m.ppf(.99),100)
# plt.plot(x,m.pdf(x), 'k-', lw=2)
# plt.show()
# print(1)
# print(m)
# i = np.random.randint(0,100,100)
# print(i)
from models import settings

sfs = StratifiedShuffleSplit( test_size=0.1, random_state=0)

filepath = 'datasets/regression_datasets/10_Merck_Molecular_Activity_Challenge'

MERCK_FILE1 = 'Merck_Data1.npz'
MERCK_FILE2 = 'Merck_Data2.npz'

MERCK_FILE1 = np.load(os.path.join(settings.ROOT_DIR, filepath, MERCK_FILE1))
MERCK_FILE2 = np.load(os.path.join(settings.ROOT_DIR, filepath, MERCK_FILE2))

x_train1 = MERCK_FILE1.get("x_train1")
x_test1 = MERCK_FILE1.get("x_test1")
y_train1 = MERCK_FILE1.get("y_train1")
y_test1 = MERCK_FILE1.get("y_test1")


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys
