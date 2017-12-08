import numpy as np
import random as rd
import itertools as it

class BatchLoader(object):
    """docstring for BatchLoader."""
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, num_class):
        super(BatchLoader, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.num_class = num_class
        self.X_train_group = self.X_val_group = [[] for _ in range(num_class)]
        for index, value in zip(y_train, X_train):
            self.X_train_group[index].append(value)
        for index, value in zip(y_val, X_val):
            self.X_val_group[index].append(value)
        self.X_train_lenght = [len(_) for _ in self.X_train_group]
        self.X_val_lenght = [len(_) for _ in self.X_val_group]
    def make_batch(self, n=100, dat_type='train'):
        '''
        Create 2N batch of sample
        N for same categories (similar)
        N for different categories
        '''
        input1 = np.zeros((2*n,28,28,1))
        input2 = np.zeros((2*n,28,28,1))
        yout = np.concatenate([np.ones((n,1)),np.zeros((n,1))])
        if dat_type == 'train':
            use_dat = self.X_train_group
            use_lenght = self.X_train_lenght
        else:
            use_dat = self.X_val_group
            use_lenght = self.X_val_lenght
        for i in range(n):
            cate = np.random.randint(0,self.num_class,1)[0]
            first, second = np.random.randint(0,use_lenght[cate],2)
            input1[i], input2[i] = use_dat[cate][first], use_dat[cate][second]
        for i in range(n, 2*n):
            cate1,cate2 = rd.sample(range(self.num_class), 2)
            first = np.random.randint(0,use_lenght[cate1],1)[0]
            second = np.random.randint(0,use_lenght[cate2],1)[0]
            input1[i], input2[i] = use_dat[cate1][first], use_dat[cate2][second]
        return input1,input2,yout

    def do_test(self,sess,corr_pred,batch_size=100):
        '''
        Test the model!!!
        match all pairs from 10000 => 5 x 10^7 pairs would be fine?
        '''
        test_size = len(self.X_test)
        all_pair = it.combinations(list(range(test_size)),2)
        num_count = 0
        input1 = input2 = np.zeros((batch_size,28,28,1))
        expected = np.zeros((batch_size,1))
        num_corr = 0
        for x,y in all_pair:
            input1[num_count] = self.X_test[x]
            input2[num_count] = self.X_test[y]
            expected[num_count] = 1 if self.y_test[x] == self.y_test[y] else 0
            num_count += 1
            if num_count % batch_size == 0:
                feed_dict = {X1:input1,X2:input2,y:expected}
                num_corr += sess.run(corr_pred, feed_dict=feed_dict)
                num_count = 0
        return num_corr / (test_size * (test_size-1)/2)
