import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class DecisionStump:
    art_data: bool
    tau: Optional[float] = None
    n_init: Optional[int] = None
    n: Optional[int] = None
    
    def generate_data(self, n, tau):
        x = np.random.uniform(low=-0.5, high=0.5, size=n)
        y = [self.sign(value) for value in x]
        for idx, label in enumerate(y):
            if np.random.random() < tau:
                y[idx] = -label
                
        return np.array(x), np.array(y)
    
    def fit(self, x, y):
        self.ans_1d = []
        self.ans_multi_d = []
        # self.test = []
        
        if self.art_data:
            # x_out, y_out = self.generate_data(100000, self.tau)
            # x_out = np.reshape(x_out, (len(x_out), 1))
            # y_out = np.reshape(y_out, (len(y_out), 1))
            for i in range(self.n_init):
                # artificial data
                self.x, self.y = self.generate_data(self.n, self.tau)
                self.x = np.reshape(self.x, (len(self.x), 1))
                self.y = np.reshape(self.y, (len(self.y), 1))
                
                # sample number and dimension of training data
                self.n = self.x.shape[0]
                self.d = self.x.shape[1]
                
                if self.d == 1:
                    ans1, ans2, ans3, ans4 = self.run_DS_1D(self.x, self.y)
                    self.ans_1d += [ans1]
                    # self.test += [self.error(x_out, y_out, ans4, ans2) - ans3]
                    
                else:
                    # won't use in this hw
                    self.ans_multi_d += self.run_DS_multiD(self.x, self.y)
        else:
            self.x = np.array(x)
            self.y = np.array(y)
            
            # sample number and dimension of training data
            self.n = self.x.shape[0]
            self.d = self.x.shape[1]
            
            result = self.run_DS_multiD(self.x, self.y)
            
            self.best_of_best_ein = result[0]
            self.worst_of_best_ein = result[1]
            self.best_of_best_theta = result[2]
            self.worst_of_best_theta = result[3]
            self.best_of_best_s = result[4]
            self.worst_of_best_s = result[5]
            self.best_of_best_dim = result[6]
            self.worst_of_best_dim = result[7]
        
        return self
    
    def error(self, x, y, s, theta):
        err = 0
        for idx, data in enumerate(x):
            if y[idx] != s * self.sign(data - theta):
                err += 1
            
        return  err/len(x)
    
    def run_DS_1D(self, x, y):
        x_n_sorted = sorted(x)
        ninf = -1e15
        thetas = np.array([ninf])
        # generate theta
        for i in range(self.n-1):
            thetas = np.append(thetas, (x_n_sorted[i]+x_n_sorted[i+1])/2)
        thetas = np.array(thetas)
        
        direction = [1, -1]
        e_best = 1
        theta_best = 0
        
        # calculate error and record best
        for s in direction:
            for theta in thetas:
                wrong = self.error(x, y, s, theta)
                if wrong < e_best:
                    e_best = wrong
                    theta_best = theta
                    s_best = s
                    
        if self.art_data:
            e_out = self.error_out(theta_best)
        else:
            # dummy eout, when we use real data, eout can be computed from test data
            e_out = 1
        e_in = e_best
        
        return e_out - e_in, theta_best, e_in, s_best
    
    def run_DS_multiD(self, x, y):
        best_thetas = []
        best_eins = []
        best_ss = []
        
        y = np.reshape(y, (len(y), 1))
        
        for dim in range(self.d):
            x_dim = np.reshape(x[:,dim], (len(x[:,dim]), 1))
            result = self.run_DS_1D(x_dim, y)
            best_thetas += [result[1]]
            best_eins += [result[2]]
            best_ss += [result[3]]
            
        best_of_best_theta = best_thetas[np.argmin(best_eins)]
        worst_of_best_theta = best_thetas[np.argmax(best_eins)]
        
        best_of_best_ein = best_eins[np.argmin(best_eins)]
        worst_of_best_ein = best_eins[np.argmax(best_eins)]
        
        best_of_best_s = best_ss[np.argmin(best_eins)]
        worst_of_best_s = best_ss[np.argmax(best_eins)]
        
        best_of_best_dim = np.argmin(best_eins)
        worst_of_best_dim = np.argmax(best_eins)
        
        return best_of_best_ein, worst_of_best_ein, best_of_best_theta, worst_of_best_theta, best_of_best_s, worst_of_best_s, best_of_best_dim, worst_of_best_dim
    
    def error_out(self, theta):
        return min(np.abs(theta), 0.5) * (1 - 2 * self.tau) + (self.tau)
    
    def predict(self, x, y, best=None):
        predict_label = []
        wrong = 0
        if not best:
            dim = self.worst_of_best_dim
            s = self.worst_of_best_s
            x_dim = np.reshape(x[:,dim], (len(x), 1))
            for idx, data in enumerate(x_dim):
                predict_label += [int(s*self.sign(data-self.worst_of_best_theta))]
                if y[idx] != s*self.sign(data-self.worst_of_best_theta):
                    wrong += 1
                eout = wrong/len(y)
        else:
            dim = self.best_of_best_dim
            s = self.best_of_best_s
            x_dim = np.reshape(x[:,dim], (len(x), 1))
            for idx, data in enumerate(x_dim):
                predict_label += [s*self.sign(data-self.best_of_best_theta)]
                if y[idx] != s*self.sign(data-self.best_of_best_theta):
                    wrong += 1
                eout = wrong/len(y)
        
        return predict_label, eout

    def sign(self, value):
        """
        The sign function

        Args:
            value (float): the dot value from PLA iteration

        Returns:
            int: if value > 0 will be 1 otherwise -1
        """
        if value > 0:
            return 1
        else:
            return -1
            