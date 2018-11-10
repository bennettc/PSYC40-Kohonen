import numpy as np

class KohonenNetwork(object):
	
	def __init__(x, y, input_length, learning_rate=0.5, 
		radius=1.0 ,random_seed=None):
		
		self.x = x
		self.y = y
		self.input_length = input_length
		self.radius = radius
		self.learning_rate = learning_rate

		self.weights = np.random.rand(x, y, input_length)

	def train(data, num_iterations):
		if len(data.shape) != 2 or data.shape[1] != input_len:
			#throw exception???
		for n in range(num_iterations):
			for x_vector in data:
				bmu = getBMU(x_vector)
				updateWeights(bmu, x_vector)

	def getBMU(x_vector):
		#produces an x by y array of all weights Euclidean distances
		#to given vector
		dist = numpy.linalg.norm(x_vector - self.weights, axis=2)
		#returns index (as a tuple) of minimum distance
		return np.unravel_index(dist.argmin(), dist.shape)

	def updateWeights(bmu, x_vector):
		for i in range(x):
			for j in range(y):
				delta_w = (neighbor(bmu, x_vector) * self.learning_rate) * 
					np.subtract(x_vector, w[i,j])

	def neighbor(index, center):
		p_dist = (index[0] - center[0])**2 + (index[1] - center[1])**2
		if p_dist <= self.radius:
			return 1
		return 0