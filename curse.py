import numpy as np
import time

class dist_point:
    def __init__(self, dist, index):
        self.dist = dist
        self.index = index

def random_array(dimensions):
    return 2*np.random.rand(1000, dimensions) - 1 

def distance_from_origin(vector):
    b = np.zeros(vector.shape)
    return np.linalg.norm(vector-b)

def fractional_distance(array):
    count = 0
    for row in array:
        if distance_from_origin(row) < 1:
            count += 1
    return float(count)/(array.shape[0])

def knn_distance(matrix, query, knum):
    dists = []
    index = 0
    for row in matrix:
        dists.append( dist_point( distance( row, query) , index ) )
        index += 1
    dists = sorted( dists, key=lambda dist_point: dist_point.dist) 
    return dists[0].dist

def distance(a_point, b_point):
     return np.linalg.norm(a_point-b_point)

def random_query_pts(dimensions):
    return 2*np.random.rand(100, dimensions) - 1

def average_distance(rand_arr, rand_points):
    distance = 0
    for row in rand_points:
        distance += knn_distance(rand_arr, row, 1)
    return float(distance) / 100
    

if __name__ == "__main__":
    start_time = time.time()
    dimensions = 1
    while dimensions < 16:
        rand_arr = random_array( dimensions)
        y = fractional_distance( rand_arr)
        rand_pts = random_query_pts( dimensions)
        avg_dist = average_distance( rand_arr, rand_pts)
        print "Dimension - "+ str(dimensions)
        print "Average fractional distance from origin - " + str(y)
        print "Average distance from nearest neighbor - "+ str(avg_dist)
        print ""
        dimensions += 1
    print time.time() - start_time, "seconds"

