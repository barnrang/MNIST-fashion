import numpy as np
import random as rd

class BatchLoader(object):
    """docstring for BatchLoader."""
    def __init__(self, X_train, y_train, X_val, y_val, num_class):
        super(BatchLoader, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_class = num_class
        self.X_train_group = self.X_val_group = [[] for _ in range(num_class)]
        for index, value in zip(y_train, X_train):
            self.X_train_group[index].append(value)
        for index, value in zip(y_val, X_val):
            self.X_val_group[index].append(value)
        self.X_train_lenght = [len(_) for _ in X_train_group]
        self.X_val_lenght = [len(_) for _ in X_val_group]
    def make_batch(self, n=100):
        '''
        Create 2N batch of sample
        N for same categories (similar)
        N for different categories
        '''
        input1 = np.zeros((2*n,28,28,1))
        input2 = np.zeros((2*n,28,28,1))
        yout = np.concatenate([np.ones((n,1)),np.zeros((n,1))])
        for i in range(n):
            cate = np.random.randint(0,self.num_class,1)[0]
            first, second = np.random.randint(0,self.X_train_lenght[cate],2)
            input1[i], input2[i] = self.X_train_group[cate][first], self.X_train[cate][second]
        for i in range(n, 2*n):
            cate1,cate2 = rd.sample(range(self.num_class), 2)
            first = np.random.randint(0,self.X_train_lenght[cate1],1)[0]
            second = np.random.randint(0,self.X_train_lenght[cate2],1)[0]
            input1[i], input2[i] = self.X_train_group[cate1][first], self.X_train[cate2][second]
        return input1,input2,yout

    def make_val(self,n=500):
        '''
        Create 2N batch of validation
        N for same categories (similar)
        N for different categories
        '''
        return None
    def do_test(self,model,acc):
        return None
