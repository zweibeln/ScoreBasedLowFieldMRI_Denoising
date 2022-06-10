import os
import sys
DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.insert(0, DIR)
from configs.imports import *
class EMA():
	def __init__(self, mu):
		self.mu = mu
		self.shadow = {}

	def register(self, name, val):
		self.shadow[name] = val.clone()

	def __call__(self, name, x):
		assert name in self.shadow
		new_average = self.mu * x + (1-self.mu) * self.shadow[name]
		self.shadow[name] = new_average.clone()
		return new_average