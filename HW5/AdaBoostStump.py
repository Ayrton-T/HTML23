import numpy as np
from dataclasses import dataclass

@dataclass
class AdaBoostStump():
    n_init: int
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.alpha, self.little_g = self.run_adaboost_ds(self.x, self.y, self.n_init)
        
        self.little_g_errors = []
        # collect 0/1 error for little_g
        for g in self.little_g:
            self.little_g_errors += [self.error(x[:,g[1]], y, g[0], g[2])]
        self.min_g_error = min(self.little_g_errors)
        self.max_g_error = max(self.little_g_errors)
        
        # compute big G error
        self.big_G_error_in = self.ada_error(x, y, self.alpha, self.little_g)
            
        return self
    
    def run_adaboost_ds(self, x, y, n_init):
        weight = np.full((len(x),), 1/len(x))
        little_g = []
        alpha = []
        best_error = 1e15
        for _ in range(n_init):
            s, dim, theta, error = self.decision_stump(x, y, weight)
            # print(error)
            # renew weights
            little_g.append([s, dim, theta])
            scaling_factor = np.sqrt((1-error)/error)
            alpha.append(np.log(scaling_factor))
            
            if best_error > error:
                best_error = error
            
            # update weights
            for i in range(len(y)):
                if y[i] != s*self.sign(x[i][dim]-theta):
                    weight[i] *= scaling_factor
                else:
                    weight[i] /= scaling_factor
            
        return alpha, little_g
    
    def decision_stump(self, x, y, weight):
        weight_sum = np.sum(weight)
        # print(f'weight sum: {weight_sum}')
        
        dim_best_error = np.zeros(x.shape[1])
        dim_best_theta = np.zeros(x.shape[1])
        dim_best_s = np.zeros(x.shape[1])
        for dim in range(x.shape[1]):
            sort_idx = np.argsort(x[:,dim])
            x_sorted = x[sort_idx][:,dim]
            y_sorted = y[sort_idx]
            weight_sorted = weight[sort_idx]
            
            # calculate the theta = ninf condition
            # when theta is ninf, s = +1 every label equal to -1 is wrong, s = -1 label = +1 is wrong
            dp_pos = np.zeros(len(x))
            # init thetas
            thetas = [-1e15] + [(x_sorted[i]+x_sorted[i+1])/2 for i in range(x.shape[0]-1)]
            thetas = np.array(thetas)
            # print(thetas)
            cnt = 0
            for i, theta in enumerate(thetas):
                # init -inf error
                if i == 0:
                    dp_pos[i] = np.sum(weight_sorted[np.where(y_sorted == -1)])
                else:
                    if theta == x_sorted[i]:
                        dp_pos[i] = dp_pos[i-1]
                        cnt += 1
                    else:
                        dp_pos[i] = dp_pos[i-1] + np.sum([weight_sorted[i-1-j] * y_sorted[i-1-j] for j in range(cnt+1)])
                        cnt = 0
            
            dp_neg = (weight_sum - dp_pos) / weight_sum
            dp_pos = dp_pos / weight_sum
            
            pos_min_idx = np.argmin(dp_pos)
            neg_min_idx = np.argmin(dp_neg)
            
            if dp_pos[pos_min_idx] <= dp_neg[neg_min_idx]:
                dim_best_s[dim] = 1
                dim_best_error[dim] = dp_pos[pos_min_idx]
                dim_best_theta[dim] = thetas[pos_min_idx]
            else:
                dim_best_s[dim] = -1
                dim_best_error[dim] = dp_neg[neg_min_idx]
                dim_best_theta[dim] = thetas[neg_min_idx]
            
        best_dim = np.argmin(dim_best_error)
        best_theta = dim_best_theta[best_dim]
        best_s = dim_best_s[best_dim]
        best_error = dim_best_error[best_dim]
        
        return best_s, best_dim, best_theta, best_error
    
    def bigG(self, x, alpha, little_g):
        temp = 0
        for idx, a in enumerate(alpha):
            s = little_g[idx][0]
            dim = little_g[idx][1]
            theta = little_g[idx][2]
            temp += a * s * self.sign(x[dim] - theta)
            
        return self.sign(temp)
    
    def ada_error(self, x, y, alpha, little_g):
        error = 0
        for idx, data in enumerate(x):
            if self.bigG(data, alpha, little_g) != y[idx]:
                error += 1
    
        return error/len(y)
    
    def error(self, x, y, s, theta):
        err = 0
        for idx, data in enumerate(x):
            if y[idx] != s * self.sign(data - theta):
                err += 1
            
        return  err/len(x)
    
    def calculate_eout(self, x, y):
        big_G_error_out = self.ada_error(x, y, self.alpha, self.little_g)
        return big_G_error_out
    
    def sign(self, value):
        if value >= 0:
            return 1
        else:
            return -1