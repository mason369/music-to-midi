import unittest
import torch
import numpy as np
from model.ops import minmax_normalize


class TestMinMaxNormalize(unittest.TestCase):

    def test_minmax_normalize(self):
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        x_norm = minmax_normalize(x)
        x_norm_expected = torch.tensor([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]])

        np.testing.assert_almost_equal(x_norm.numpy(), x_norm_expected.numpy(), decimal=2)


if __name__ == '__main__':
    unittest.main()
