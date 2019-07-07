# Unittest for light_atom_interaction.py module
import sys
import unittest
import env
from light_atom_interaction import doppler_shift, lorentzian_probability



class Light_Atom_Interaction_Test(unittest.TestCase):

    def testDopplerShift(self):
        self.assertEqual(doppler_shift(100, 100, 100, 5E14, 0,0,1, 670.977E-9), 4.999990635766491E14, msg=None)
        self.assertEqual(doppler_shift(0, 0, 1000, 5E14, 0,0,1, 670.977E-9), 4.99990635766491E14, msg=None)
        self.assertEqual(doppler_shift(1000, 0, 0, 5E14, 0,0,1, 670.977E-9), 5E14, msg=None)
        self.assertEqual(doppler_shift(0, 1000, 0, 5E14, 0,0,1, 670.977E-9), 5E14, msg=None)
		
    def testLorentzianProbability(self):
        self.assertLessEqual(lorentzian_probability(1, 3, 2, 1, 1, 1), 1/66, msg=None)
        self.assertLessEqual(lorentzian_probability(1, 3, 2, 1, -1, 1), 1/64, msg=None)
        self.assertGreaterEqual(lorentzian_probability(1, 3, 2, 2, 3, 4), 4/71, msg=None)
        self.assertGreaterEqual(lorentzian_probability(1, 3, -2, 1, 1, -2), 2, msg=None)
	
if __name__ == "__main__":
    unittest.main()