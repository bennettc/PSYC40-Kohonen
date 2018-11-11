import unittest
import numpy as np

from kohonen.kohonen import KohonenNetwork

class SimpleTest(unittest.TestCase):
	def setUp(self):
		self.training_data = np.array([[1, 1], [2, 2], [2, 3], [8, 8]])
		self.resetNet()

	def resetNet(self):
		self.net = KohonenNetwork(3, 3, 2)
		self.net.weights = np.arange(18.).reshape(3, 3, 2)

	def test_BMU_1(self):
		vector = np.array([8,9])
		# vector is closest to weight at (1, 1)
		self.assertEqual(self.net.getBMU(vector), (1,1))

	def test_BMU_2(self):
		vector = np.array([16,17])
		# vector is closest to weight at (2, 2)
		self.assertEqual(self.net.getBMU(vector), (2,2))

	def test_getFeatureMap(self):
		fm = self.net.getFeatureMap(self.training_data)
		#should map [1,1] 			to weight (0,0) == [0,1]
		#			[2,2] and [2,3] to weight (0,1) == [2,3]
		#			[8,8] 			to weight (1,1) == [8,9]
		assert np.array_equal(fm[(0,0)][0], np.array([1,1]))
		assert np.array_equal(fm[(0,1)][0], np.array([2,2]))
		assert np.array_equal(fm[(0,1)][1], np.array([2,3]))
		assert np.array_equal(fm[(1,1)][0], np.array([8,8]))

		self.resetNet

	def test_Neighbor_1(self):
		self.assertEqual(self.net.neighbor((0,0), (0,1)), 1)

	def test_Neighbor_2(self):
		self.assertEqual(self.net.neighbor((0,0), (1,1)), 1)

	def test_Neighbor_3(self):
		self.assertEqual(self.net.neighbor((0,0), (0,2)), 1)

	def test_Neighbor_4(self):
		self.assertEqual(self.net.neighbor((0,0), (1,2)), 0)

	def test_checkDataShape_1(self):
		test_data = np.array([1,2])
		self.assertRaises(Exception, self.net.checkDataShape, test_data)

	def test_checkDataShape_2(self):
		test_data = np.array([[1,2,4], [1,2,3]])
		self.assertRaises(Exception, self.net.checkDataShape, test_data)

	def test_checkDataShape_3(self):
		test_data = np.array([[1,2], [1,3]])
		self.net.checkDataShape(test_data)

	def test_updateWeights(self):
		#every neuron is in a neighborhood of the center neuron, so every weight 
		#should be updated to be 
		#the current weights minus learning_rate(0.5)*current weights 
		target_weights = 0.5*self.net.weights
		self.net.updateWeights((1,1), np.array([0,0]))

		assert np.array_equal(self.net.weights, target_weights)

	# def test(self):
	# 	self.net.train(self.training_data, 10)
	# 	fm = self.net.getFeatureMap(self.training_data)