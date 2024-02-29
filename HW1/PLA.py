import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class PLA:
    M: int # PLA stop after check M randomly-picked sample correct consecutively
    n_init: int
    x_0: float
    scale: float
    
    def fit(self, x, y):
        """
        Train PLA by x and y
        Args:
            x (List(float) or ndarray): The input training vector space of samples
            y (List(int) or ndarray): The label of input vector space of samples

        Returns:
            PLA class
        """
        # add x_0 to first column
        self.x = np.insert(x, 0, self.x_0, axis=1)*self.scale
        self.y = y
        
        # the number of training samples
        self.n = len(x)
        
        # answers
        self.Ein = []
        self.updates = []
        self.w_pla = []
        self.x_0_w_pla = []
        
        # iteration
        for i in range(self.n_init):
            result = self.run_PLA(self.M)
            # collect answers from different iteration
            self.Ein += [result[0]]
            self.updates += [result[1]]
            self.w_pla += [result[2]]
            self.x_0_w_pla += [result[3]]
        
        # turn answers list into numpy array
        self.Ein = np.array(self.Ein)
        self.updates = np.array(self.updates)
        self.w_pla = np.array(self.w_pla)
        self.x_0_w_pla = np.array(self.x_0_w_pla)
        return self
    
    def run_PLA(self, M):
        """
        Run PLA one iter with M random sample correct consecutively as termination condition
        Args:
            M (int): a PLA termination requirement, need to correct M(with random sample) consecutively times to terminate PLA

        Returns:
            float, int, list(float): the answers of HW
        """
        # if cnt == M, PLA terminate
        cnt = 0
        # count total iteration
        iter = 0
        # init w_t and updates
        w_t = np.zeros(self.x.shape[1])
        updates = 0.0
        
        while(cnt != M):
            # pick sample randomly, generate one index randomly befor loop
            sample_idx = np.random.randint(self.n, size=1)
            x = self.x[sample_idx].flatten()
            y = self.y[sample_idx].flatten()
            
            iter += 1
            cnt += 1
            if self.sign(np.dot(w_t, x)) != y:
                updates += 1
                cnt = 0
                w_t += x * y

        # init error and calculate Ein
        error = 0.0
        for i in range(self.n):
            x = self.x[i].flatten()
            y = self.y[i].flatten()
            if self.sign(np.dot(w_t, x)) != y:
                error += 1
        Ein = error/self.n
        return Ein, int(updates), w_t, self.x_0 * w_t[0]

    def sign(self, value):
        """
        The sign function

        Args:
            value (float): the dot value from PLA iteration

        Returns:
            int: if value >= 0 will be 1 otherwise -1
        """
        if value >= 0:
            return 1
        else:
            return -1