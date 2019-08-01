# Unittest for light_atom_interaction.py module
import sys
import unittest
import env
from light_atom_interaction import doppler_shift, lorentzian_probability, laser_intensity_gauss, scatter_rate


class Light_Atom_Interaction_Test(unittest.TestCase):

    def testDopplerShift(self):
        self.assertEqual(doppler_shift(100, 100, 100, 5E14, 0,0,1, 670.977E-9), 4.999990635766491E14, msg=None)
        self.assertEqual(doppler_shift(0, 0, 1000, 5E14, 0,0,1, 670.977E-9), 4.99990635766491E14, msg=None)
        self.assertEqual(doppler_shift(1000, 0, 0, 5E14, 0,0,1, 670.977E-9), 5E14, msg=None)
        self.assertEqual(doppler_shift(0, 1000, 0, 5E14, 0,0,1, 670.977E-9), 5E14, msg=None)
		
    def testLorentzianProbability(self):
        self.assertLessEqual(lorentzian_probability(1, 3, 2, 1, 1, 1), 0.01, msg=None)
        self.assertLessEqual(lorentzian_probability(1, 3, 2, 1, 2, 1), 6E-4, msg=None)
        self.assertGreaterEqual(lorentzian_probability(1, 3, 2, 2, 3, 4), 0.005, msg=None)
        self.assertGreaterEqual(lorentzian_probability(1, 3, -2, 1, 1, 2), 0.001, msg=None)
        self.assertLessEqual(lorentzian_probability(1, 3, 2E6, 1, 1, 1), 1.0E-14, msg=None)
        self.assertLessEqual(lorentzian_probability(1, 3, 2E6, 1, 2, 1), 1.0E-14, msg=None)
        self.assertGreaterEqual(lorentzian_probability(1, 3, 2E6, 2, 3, 4), 0.0, msg=None)
        self.assertGreaterEqual(lorentzian_probability(1, 3, -2E6, 1, 1, 2), 0.0, msg=None)

		
    def testLaserIntensityGauss(self):
        self.assertEqual(laser_intensity_gauss(0.0002, 1.0, 1.0), 1.7815252551728327, msg=None)
        self.assertEqual(laser_intensity_gauss(0.0001, 0.5, 0.1), 0.25642497617439414, msg=None)
        self.assertEqual(laser_intensity_gauss(1.0E-5, 0.1, 10.0), 21.087579151860186, msg=None)
        self.assertEqual(laser_intensity_gauss(1.0E-7, 0.01, 0.01), 0.010215416304041422, msg=None)
		
    # def testScatterRate(self):
    #     self.assertEqual(scatter_rate(, , , , ), , msg=None)
    #     self.assertEqual(scatter_rate(, , , , ), , msg=None)
    #     self.assertEqual(scatter_rate(, , , , ), , msg=None)
    #     self.assertEqual(scatter_rate(, , , , ), , msg=None)
	
if __name__ == "__main__":
    unittest.main()