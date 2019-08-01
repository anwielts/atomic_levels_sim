# Unittest for distributions.py module
import sys
import unittest
import env
from distributions import maxwell_boltzmann_distribution_v_2, maxwell_boltzmann_distribution_v_3


class DistributionsTest(unittest.TestCase):

    def testMaxwellBoltzmannV2(self):
        self.assertLessEqual(maxwell_boltzmann_distribution_v_2(0, 6.94*1.66E-27, 650), 1, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution_v_2(5000, 6.94*1.66E-27, 650), 1, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_2(0, 6.94*1.66E-27, 650), 0, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_2(5000, 6.94*1.66E-27, 650), 0, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution_v_2(0, 1.0*1E27, 1000), 1, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution_v_2(10000, 1.0*1E-27, 650), 1, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_2(1, 1, 1), 0, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_2(5000, 6.94*1.66E-27, 1E-15), 0, msg=None)
	
    def testMaxwellBoltzmannV3(self):
        self.assertLessEqual(maxwell_boltzmann_distribution_v_3(0, 6.94*1.66E-27, 650), 1, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution_v_3(5000, 6.94*1.66E-27, 650), 1, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_3(0, 6.94*1.66E-27, 650), 0, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_3(5000, 6.94*1.66E-27, 650), 0, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution_v_3(0, 1.0*1E27, 1000), 1, msg=None)
        self.assertLessEqual(maxwell_boltzmann_distribution_v_3(10000, 1.0*1E-27, 650), 1, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_3(1, 1, 1), 0, msg=None)
        self.assertGreaterEqual(maxwell_boltzmann_distribution_v_3(5000, 6.94*1.66E-27, 1E-15), 0, msg=None)
	
if __name__ == "__main__":
    unittest.main()