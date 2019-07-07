# Unittest for util.py module
import sys
import unittest
import env
from util import calculate_p_max


class Help_Functions_Test(unittest.TestCase):

    # def testRandomVelocityAssigment(self):
        # self.assertLessEqual(Help_Functions.random_velocity_assignment(), 1, msg=None)
        # self.assertLessEqual(Help_Functions.random_velocity_assignment(), 1, msg=None)
        # self.assertGreaterEqual(Help_Functions.random_velocity_assignment(), 0, msg=None)
        # self.assertGreaterEqual(Help_Functions.random_velocity_assignment(), 0, msg=None)
		
    def testCalculatePMax(self):
        self.assertLessEqual(calculate_p_max(1000, 0, 5000, 6.94*1.661E-27, 650), 1, msg=None)
        self.assertLessEqual(calculate_p_max(1000, 0, 5000, 6.94*1.661E-27, 650), 1, msg=None)
        self.assertGreaterEqual(calculate_p_max(1000, 0, 5000, 6.94*1.661E-27, 650), 0, msg=None)
        self.assertGreaterEqual(calculate_p_max(1000, 0, 5000, 6.94*1.661E-27, 650), 0, msg=None)
	
	
if __name__ == "__main__":
    unittest.main()