import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Regression:
    algo:Optional[str] = None
    n_init:Optional[int] = None
    learning_rate:Optional[float] = None
    Q: Optional[int] = None
    
    def fit(self, x, y):
        self.x = np.insert(x, 0, 1, axis=1)
        self.y = y
        self.n = x.shape[0]
        
        self.Ein_linear_sgd = []
        self.Ein_logistic_sgd = []
        self.Ein_logistic_sgd_wlin = []
        
        if self.algo == 'linear regression':
            self.Ein = self.run_regression(self.x, self.y)[0]
        elif self.algo == 'linear sgd':
            for i in range(self.n_init):
                result = self.run_sgd_linear(self.x, self.y)[0]
                self.Ein_linear_sgd += [result]
        elif self.algo == 'logistic sgd':
            for i in range(self.n_init):
                result = self.run_sgd_logistic(self.x, self.y)[0]
                self.Ein_logistic_sgd += [result]
        elif self.algo == 'logistic sgd wlin':
            for i in range(self.n_init):
                result = self.run_sgd_logistic_wlin(self.x, self.y)[0]
                self.Ein_logistic_sgd_wlin += [result]         
                
        return self
    
    def w800_ein_eout(self, x, y, x_test, y_test):
        self.x = np.insert(x, 0, 1, axis=1)
        self.y = y
        self.n = x.shape[0]
        
        self.x_test = np.insert(x_test, 0, 1, axis=1)
        self.y_test = y_test
        
        self.ans17 = []
        
        for i in range(self.n_init):
            w800 = self.run_sgd_logistic_wlin(self.x, self.y)[1]
            self.ans17 += [np.abs(self.Error01(w800, self.x, self.y) - self.Error01(w800, self.x_test, self.y_test))]
        
        return self
    
    def wlin_ein_eout(self, x, y, x_test, y_test):
        self.x = np.insert(x, 0, 1, axis=1)
        self.y = y
        self.n = x.shape[0]
        
        self.x_test = np.insert(x_test, 0, 1, axis=1)
        self.y_test = y_test
        
        self.ans18 = 0
        
        wlin = self.run_regression(self.x, self.y)[1].flatten()
        self.ans18 = np.abs(self.Error01(wlin, self.x, self.y) - self.Error01(wlin, self.x_test, self.y_test))
        
        return self
    
    def Q_ein_eout(self, x, y, x_test, y_test):
        self.x = np.insert(x, 0, 1, axis=1)
        self.y = y
        self.n = x.shape[0]
        
        self.x_test = np.insert(x_test, 0, 1, axis=1)
        self.y_test = y_test
        
        self.ans_last2 = []
        
        self.transform_x = self.Q_transform(self.Q, self.x)
        self.transform_x_test = self.Q_transform(self.Q, self.x_test)
        
        wlin = self.run_regression(self.transform_x, self.y)[1].flatten()
        self.ans_last2 = np.abs(self.Error01(wlin, self.transform_x, self.y) - self.Error01(wlin, self.transform_x_test, self.y_test))

        return self
    
    def Q_transform(self, Q, x):
        original_x = x[:, 1:]
        new_x = x.copy()
        for i in range(2, Q+1):
            new_x = np.hstack((new_x, original_x**i))
        # print(new_x.shape)
        return new_x
    
    def run_regression(self, x, y):
        w = self.pseudo_inverse(x, y)
        Ein = self.Ein_sqrt(w, x, y)
        
        return Ein, w

    
    def run_sgd_linear(self, x, y):
        eta = self.learning_rate
        # w0 init
        w_t = np.zeros(x.shape[1])
        
        for i in range(800):
            sample_idx = np.random.randint(self.n, size=1)
            x_random = x[sample_idx].flatten()
            y_random = y[sample_idx].flatten()
            
            w_t += eta * 2 * (y_random-np.dot(w_t, x_random)) * x_random
            
        Ein = self.Ein_sqrt(w_t, x, y)    
        return Ein, w_t
    
    def run_sgd_logistic(self, x, y):
        eta = self.learning_rate
        w_t = np.zeros(x.shape[1])
        
        for i in range(800):
            sample_idx = np.random.randint(self.n, size=1)
            x_random = x[sample_idx].flatten()
            y_random = y[sample_idx].flatten()
            
            w_t += eta * self.sigmoid(-y_random*np.dot(w_t, x_random))*(y_random * x_random)
            
        return self.Ein_ce(w_t, x, y), w_t
    
    def run_sgd_logistic_wlin(self, x, y):
        eta = self.learning_rate
        w_t = self.pseudo_inverse(x, y).flatten()
        
        for i in range(800):
            sample_idx = np.random.randint(self.n, size=1)
            x_random = x[sample_idx].flatten()
            y_random = y[sample_idx].flatten()
            
            w_t += eta * self.sigmoid(-y_random*np.dot(w_t, x_random))*(y_random * x_random)
            
        return self.Ein_ce(w_t, x, y), w_t
    
    def pseudo_inverse(self, x, y):
        return np.dot(np.linalg.pinv(x), y)
    
    def Ein_sqrt(self, w, x, y):
        return np.mean([(np.dot(w.flatten(), x[i]) - y[i])**2 for i in range(len(y))])       
    
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    # def Ein_ce(self, w, x, y):
    #     return np.mean(np.log(1+np.exp(-y*np.dot(x,w))))
    
    def Ein_ce(self, w, x, y):
        return np.mean(np.array([np.log(1+np.exp(-y[i].flatten()*np.dot(x[i].flatten(),w))) for i in range(len(y))]))
    
    def Error01(self, w, x, y):
        error = 0.0
        for i in range(len(y)):
            sample_x = x[i].flatten()
            sample_y = y[i].flatten()
            if self.sign(np.dot(w, sample_x)) != sample_y:
                error += 1
        return error/len(y)
    
    def sign(self, value):
        if value >= 0:
            return 1
        else:
            return -1
    