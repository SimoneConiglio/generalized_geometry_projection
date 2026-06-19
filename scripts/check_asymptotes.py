import sys
import os
sys.path.append('Uno/build')
import unopy
import numpy as np

# Mock model
class SimpleModel:
    def __init__(self):
        self.num_vars = 2
        self.num_cons = 0
    def evaluate(self, x, obj_multiplier, multipliers):
        return x[0]**2 + x[1]**2, [], [2*x[0], 2*x[1]], []

model = unopy.Model(SimpleModel(), [0.0]*2, [1.0]*2, [], [])
solver = unopy.UnoSolver()
solver.set_preset("filtersmma")
solver.set_option("max_iterations", 5)
solver.set_option("logger", "DEBUG")
result = solver.optimize(model)
