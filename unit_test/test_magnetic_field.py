# Unittest for magnetic_field.py module
import sys
import unittest
import env
from magnetic_field import magnetic_field_function, magnetic_field_polyfit


class Magnetic_Field_Test(unittest.TestCase):

    # def testMagneticFieldPolyfit(self):
        # self.assertEqual(magnetic_field_polyfit(), 1, msg=None)
        # self.assertEqual(magnetic_field_polyfit(), 1, msg=None)
        # self.assertEqual(magnetic_field_polyfit(), 0, msg=None)
        # self.assertEqual(magnetic_field_polyfit(), 0, msg=None)
		
    def testMagneticFieldFunction(self):
        self.assertEqual(magnetic_field_function([[2.0, -2.0, 1.0], 0.002, 3], 0.4), 0.52, msg=None)
        self.assertEqual(magnetic_field_function([[2.0, -2.0, 1.0], 0.002, 3], -0.4), 2.12, msg=None)
        self.assertEqual(magnetic_field_function([[2.0, -2.0, 1.0], 0.002, 3], 0.0), 1.0, msg=None)
        self.assertEqual(magnetic_field_function([[2.0, -2.0, 1.0], 0.002, 3], 1.0), 1.0, msg=None)
	
if __name__ == "__main__":
    unittest.main()