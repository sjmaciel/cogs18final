# import sys
# sys.path.append("..")

from lplot import abline
#
# # def plot_test
# #
# #     self.assertRaises(ValueError,abline)
#
def test_abline():
    print ("hey")
    assert abline([2],[3]) == None
