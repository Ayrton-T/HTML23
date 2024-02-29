# %%
import numpy as np
from libsvm.svmutil import *

# %%
C = 1
def label_transform(y, target_label):
    return list(map(lambda x: 2*x-1, np.array(y) == target_label))


# %%
y, x = svm_read_problem('./letter.scale.tr')
p11_y = label_transform(y, 1)
m = svm_train(p11_y, x, '-s 0 -t 0 -c 1')

# %%
