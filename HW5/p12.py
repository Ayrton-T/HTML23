import numpy as np
from libsvm.svmutil import *

def label_transform(y, target_label):
    return list(map(lambda x: 2*x-1, np.array(y) == target_label))

y, x = svm_read_problem('./letter.scale.tr')

C = 1
Q = 2

ein_dict = {}
for i in range(2, 7):
    p12_y = label_transform(y, i)
    m = svm_train(p12_y, x, f'-s 0 -t 1 -c {C} -d {Q} -g 1 -r 1')
    p_label, p_acc, p_val = svm_predict(p12_y, x, m)
    ein = np.mean(np.array(p_label) != np.array(p12_y))
    ein_dict[i] = ein
    
ans = max(ein_dict, key=ein_dict.get)
print(ans)