import numpy as np
import math

class KohonenNetwork(object):
	def __init__(self, x, y, input_length, learning_rate=0.5, \
		radius=2.0 ,random_seed=None):
		
		self.x = x
		self.y = y
		self.input_length = input_length
		self.radius = radius
		self.learning_rate = learning_rate

		self.weights = np.random.rand(x, y, input_length)

	def train(self, data, num_iterations):
		self.checkDataShape(data)

		for n in range(num_iterations):
			for x_vector in data:
				bmu = self.getBMU(x_vector)
				self.updateWeights(bmu, x_vector)

	def getBMU(self, x_vector):
		#produces an x by y array of all weights Euclidean distances
		#to given vector
		dist = np.linalg.norm(x_vector - self.weights, axis=2)
		#returns index (as a tuple) of minimum distance
		return np.unravel_index(dist.argmin(), dist.shape)

	def updateWeights(self, bmu, x_vector):
		for i in range(self.x):
			for j in range(self.y):
				delta_w = (self.neighbor(bmu, x_vector)*self.learning_rate) * \
					np.subtract(x_vector, self.weights[i,j])

				self.weights[i,j] = np.add(self.weights[i,j], delta_w)

	def neighbor(self, index, center):
		p_dist = (index[0] - center[0])**2 + (index[1] - center[1])**2
		if math.sqrt(p_dist) <= self.radius:
			return 1
		return 0

	def getFeatureMap(self, data):
		self.checkDataShape(data)

		featureMap = {}
		for x_vector in data:
			winner = self.getBMU(x_vector)
			if not winner in featureMap:
				featureMap[winner] = []
			featureMap[winner].append(x_vector) 
		return featureMap

	def checkDataShape(self, data):
		if len(data.shape) != 2 or data.shape[1] != self.input_length:
			raise Exception('Training data must be formatted as a 2-D numpy \
				array where the second dimension is equal to set input_length')