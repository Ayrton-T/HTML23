import numpy as np
from libsvm.svmutil import *

def label_transform(y, target_label):
    return list(map(lambda x: 2*x-1, np.array(y) == target_label))

def train_test_spilt(x, y):
    rng = np.random.default_rng()
    numbers = rng.choice(len(x), size=len(x), replace=False)
    x, y = np.array(x), np.array(y)
    # 200 tranining samples
    x_eval = x[numbers[:200]]
    y_eval = y[numbers[:200]]
    # rest evaluation samples
    x_train = x[numbers[200:]]
    y_train = y[numbers[200:]]
    
    return x_train, x_eval, y_train, y_eval


y, x = svm_read_problem('./letter.scale.tr')
x = np.array(x)
gammma = [0.1, 1, 10, 100, 1000]
p16_y = np.array(label_transform(y, 7))

ans = {
        0.1:0,
          1:0,
         10:0,
        100:0,
       1000:0
       }

for _ in range(500):
    x_train, x_eval, y_train, y_eval = train_test_spilt(x, p16_y)
    e_eval = {}
    for g in gammma:
        m = svm_train(y_train, x_train, f'-s 0 -t 2 -c 0.1 -g {g} -q')
        p_label, p_acc, p_val = svm_predict(y_eval, x_eval, m, '-q')
        e_eval[g] = 1 - p_acc[0] * 0.01
    print(e_eval)
    ans[min(e_eval, key=e_eval.get)] += 1
    
print(ans)
print(max(ans, key=ans.get))