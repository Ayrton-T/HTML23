import numpy as np
from libsvm.svmutil import *

def label_transform(y, target_label):
    return list(map(lambda x: 2*x-1, np.array(y) == target_label))


C = [0.01, 0.1, 1, 10, 100]
y, x = svm_read_problem('./letter.scale.tr')
p14_y = label_transform(y, 7)
y_test, x_test = svm_read_problem('./letter.scale.t')
p14_y_test = label_transform(y_test, 7)

eout_dict = {}
for c in C:
    m = svm_train(p14_y, x, f'-s 0 -t 2 -c {c} -g 1')
    p_label, p_acc, p_val = svm_predict(p14_y_test, x_test, m)
    eout_dict[c] = np.mean(np.array(p_label) != np.array(p14_y_test))

print(eout_dict)
ans = min(eout_dict, key=eout_dict.get)
print(ans)