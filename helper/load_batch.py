import numpy as np

class BatchLoader(object):
    """docstring for BatchLoader."""
    def __init__(self, X_train, y_train, X_val, y_val):
        super(BatchLoader, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    def make_batch(self, n):
        '''
        Create 2N batch of sample
        N for same categories (similar)
        N for different categories
        '''
        return None

    def make_val(self,n):
        '''
        Create 2N batch of validation
        N for same categories (similar)
        N for different categories
        '''
        return None
    def do_test(self,model,acc):
        return None
