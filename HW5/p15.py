import numpy as np
from libsvm.svmutil import *

def label_transform(y, target_label):
    return list(map(lambda x: 2*x-1, np.array(y) == target_label))


gammma = [0.1, 1, 10, 100, 1000]
y, x = svm_read_problem('./letter.scale.tr')
p15_y = label_transform(y, 7)
y_test, x_test = svm_read_problem('./letter.scale.t')
p15_y_test = label_transform(y_test, 7)

eout_dict = {}
for g in gammma:
    m = svm_train(p15_y, x, f'-s 0 -t 2 -c 0.1 -g {g}')
    p_label, p_acc, p_val = svm_predict(p15_y_test, x_test, m)
    eout_dict[g] = np.mean(np.array(p_label) != np.array(p15_y_test))

print(eout_dict)
ans = min(eout_dict, key=eout_dict.get)
print(ans)