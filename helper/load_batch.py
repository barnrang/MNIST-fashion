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

    def make_batch_oneshot(self,test_im):
        '''
        Input: 1-image from test
        Output: random an image from all class to test neighborhood
        '''
        chosen = [rd.choice(_) for _ in self.X_train_group]
        test_stack = [test_im for _ in range(self.num_class)]
        return test_stack, chosen

    def do_test_oneshot(self, sess, dist,X1,X2,repeat=3):
        test_size = len(self.X_test)
        count_correct = 0
        for i in range(test_size):
            all_pred = np.zeros(self.num_class)
            for j in range(repeat):
                first, second = self.make_batch_oneshot(self.X_test[i])
                tmp = sess.run(dist, feed_dict={X1:first, X2:second})
                # print(tmp[:,0])
                all_pred += tmp[:,0]
            count_correct += 1 if np.argmax(all_pred) == self.y_test[i] else 0
        return count_correct/test_size
