# Unittest for distributions.py module
import sys
import unittest
import env
from distributions import maxwell_boltzmann_distribution


class DistributionsTest(unittest.TestCase):

    def testMaxwellBoltzmann(self):
        self.assertLessEqual(maxwell_boltzmann_distribution(0, 6.94*1.66E-27, 650), 1, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution(5000, 6.94*1.66E-27, 650), 1, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution(0, 6.94*1.66E-27, 650), 0, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution(5000, 6.94*1.66E-27, 650), 0, msg=None)
	
	
if __name__ == "__main__":
    unittest.main()