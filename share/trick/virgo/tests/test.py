#!/usr/bin/env python3

"""Unit test script to test VIRGO module"""

import os, sys, pdb
import unittest, argparse

import ut_VirgoActor

# Define load_tests function for dynamic loading using Nose2
def load_tests(*args):
    passed_args = locals()
    suite = unittest.TestSuite()
    suite.addTests(ut_VirgoActor.suite())
    return suite

# Local module level execution only
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Run all VIRGO unit tests.'
    )
    parser.add_argument('-v', '--visualize', action="store_true",
      help='Visualize all tests configured with this capability by setting'
      ' VIRGO_VISUALIZE_TESTS=1 before execution'
    )
    parser.add_argument('-s', '--save-images', action="store_true",
      help='Save images for all tests configured with this capability by'
      ' setting VIRGO_WRITE_TEST_IMAGES=1 before execution'
    )
    args = parser.parse_args()
    if args.visualize:
      os.environ["VIRGO_VISUALIZE_TESTS"] = "1"
    if args.save_images:
      os.environ["VIRGO_WRITE_TEST_IMAGES"] = "1"

    # Create the suite
    suites = unittest.TestSuite()
    suites.addTests(ut_VirgoActor.suite())

    # Execute all tests
    unittest.TextTestRunner(verbosity=2).run(suites)
