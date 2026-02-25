from utils.unimax_sampler.unimax_sampler import UnimaxSampler

language_character_counts = [100, 200, 300, 400, 500]
total_character_budget = 1000
num_epochs = 2

# Create the UnimaxSampler.
sampler = UnimaxSampler(language_character_counts, total_character_budget, num_epochs)

# Define the expected output. This will depend on your specific implementation of Unimax.
expected_output = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])

# Use PyTorch's allclose function to compare the computed and expected outputs.
# The absolute tolerance parameter atol specifies the maximum difference allowed for the test to pass.
self.assertTrue(torch.allclose(sampler.p, expected_output, atol=1e-6))