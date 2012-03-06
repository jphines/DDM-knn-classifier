import numpy as np
import random
import time
DIGIT_FILE = "zip.train"
SPAM_FILE = "spambase.data"
DIGIT = True

class knn:
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector
    
class dist_point:
    def __init__(self, dist, index):
        self.dist = dist
        self.index = index

class random_data_set:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

def parse_data():
    if DIGIT:
        data = open(DIGIT_FILE,"r")
    else:
        data = open(SPAM_FILE,"r")
    array = None
    vector = []
    line = data.readline()
    while line:
        if DIGIT:
            list_data = line.split()
        else:
            list_data = line.split(",")
        points = []
        for x in list_data:
            points.append( float(x) )
        if DIGIT:
            vector.append( int( points.pop(0) ) )
        else:
            vector.append( int ( points.pop() ) )
        row = np.array( points)
        if array == None:
            array = np.array( row)
        else:
            array = np.vstack( (array, row) )
        line = data.readline()
    return knn( array, vector)

def distance(a_point, b_point):
     return np.linalg.norm(a_point-b_point)

# Given training data in the N-by-D matrix X and a D-dimensional query point x,
# return the indices (of X) of the k nearest neighbors of x.
def knn_find_nearest(matrix, query, knum):
    dists = []
    index = 0
    for row in matrix:
        dists.append( dist_point( distance( row, query) , index ) )
        index += 1
    dists = sorted( dists, key=lambda dist_point: dist_point.dist) 
    count = 0
    klist = []
    while count < knum:
        klist.append( dists[count].index)
        count += 1
    return klist
    
# Given training data in the N-by-D matrix X and
# N-dimensional vector y, along with a D-dimensional query point x, return
# the predicted value yhat of the query point as the mode of the labels of
# the k nearest neighbors of x
def knn_predict(matrix, vector, query, knum):
    klist = knn_find_nearest(matrix, query, knum)
    modes = dict()
    for index in klist:
        if vector[index] in modes:
            modes[vector[index]] += 1
        else:
            modes[vector[index]] = 1
    max = -1
    max_class = -1
    iter = modes.iterkeys()
    for x in iter:
        if modes[x] > max:
            max = modes[x]
            max_class = x
    return max_class

# Given data in the N-by-D matrix X and the N=dimensional vector y, 
# partition the data into (randomized) training and testing sets, where split species
# the percentage of the data to use for training. kvec is a vector of the values of k for which to
# run the cross-validation. Return the test error for each value in kvec.
def knn_cross_validate(matrix, vector, kvector, split):
    test_error = []
    for x in kvector:
        print "Processing K value = " + str(x)
        if DIGIT == True:
            confusion_table = np.zeros( (10, 10), dtype = np.int8) 
        else:
            confusion_table = np.zeros( (2, 2), dtype = np.int8) 
        data_set = randomize_data( matrix.copy(), list(vector), split)
        test_a = data_set.test_set.matrix
        test_v = data_set.test_set.vector
        train_a = data_set.train_set.matrix
        train_v = data_set.train_set.vector
        index = 0
        correct = 0
        for row in test_a:
            prediction = int( knn_predict(train_a, train_v, row, x) )
            actual = int ( test_v[index] )
            confusion_table[actual][prediction] += 1
            if prediction == actual:
               correct += 1
            index +=1
        error = float( correct) / float( index)
        print error
        print confusion_table
        test_error.append(error)
    return test_error
        
def randomize_data(matrix, vector, split):
    numitems = matrix.shape[0]
    test_size = numitems * ( 1 - split )      #Size of test set
    test_vector = []
    count = 0
    test_matrix = None
    while count < test_size:
        row_num = random.randint( 0, (numitems-1) )
        row = matrix[row_num].copy()
        test_vector.append( vector.pop( row_num) )
        matrix = np.delete( matrix, row_num, axis = 0)
        if test_matrix != None:
            test_matrix = np.vstack( (test_matrix, row) )
        else:
            test_matrix = np.array( row)
        count += 1
        numitems -= 1
    #Create knn objects
    training = knn( matrix.copy(), list(vector) )
    test = knn( test_matrix, test_vector)
    return random_data_set(training, test)

def mean_zero_variance_one(array):
    return (array - np.mean(array)) / np.std(array)

if __name__ == "__main__":
    DIGIT = True
    start_time = time.time()
    
    digit = parse_data()
    print knn_cross_validate(digit.matrix, digit.vector, [1,2,3,4,5], .8)
    
    digit_time = time.time()
    print time.time() - start_time, "seconds"
   
    DIGIT = False
    
    spam = parse_data()
    print  knn_cross_validate(spam.matrix, spam.vector, [1,2,3,4,5], .8)
    spam_raw_time = time.time()
    print spaw_ram_time - digit_time, "seconds"
    
    spam = parse_data()
    print knn_cross_validate( mean_zero_variance_one( spam.matrix) , spam.vector, [1,2,3,4,5], .8)
    print time.time() - start_time, "seconds"
