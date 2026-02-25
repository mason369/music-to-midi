import torch
import unittest
from model.spectrogram import Melspectrogram


class TestMelspectrogram(unittest.TestCase):

    def test_melspectrogram(self):
        # Create a Melspectrogram instance with default parameters
        melspec = Melspectrogram()

        # Create a random input tensor (B, C, T) with T = 32767 samples for 2048 ms
        x = torch.randn(2, 1, 32767)

        # Compute the Melspectrogram
        y = melspec(x)

        # Check the output shape
        self.assertEqual(y.shape, (2, 256, 512))

        # Check if the output contains NaN values
        self.assertFalse(torch.isnan(y).any())

        # Check if the output contains infinite values
        self.assertFalse(torch.isinf(y).any())


if __name__ == "__main__":
    unittest.main()