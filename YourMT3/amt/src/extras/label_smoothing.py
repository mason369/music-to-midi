import torch
import numpy as np
import matplotlib.pyplot as plt

a = torch.signal.windows.gaussian(11, sym=True, std=3)
plt.plot(a)


def gaussian_smoothing(y_hot, mu=5, sigma=0.865):
    """
    y_hot: one-hot encoded array
    """
    #sigma = np.sqrt(np.abs(np.log(0.05) / ((4 - mu)**2))) / 2

    # Generate index array
    i = np.arange(len(y_hot))

    # Gaussian function
    y_smooth = np.exp(-(i - mu)**2 / (2 * sigma**2))

    # Normalize the resulting array
    y_smooth /= y_smooth.sum()
    return y_smooth, sigma


# y_ls = (1 - α) * y_hot + α / K, where K is the number of classes, alpha is the smoothing parameter

y_hot = torch.zeros(11)
y_hot[5] = 1
plt.plot(y_hot, 'b.-')

alpha = 0.3
y_ls = (1 - alpha) * y_hot + alpha / 10
plt.plot(y_ls, 'r.-')

y_gs, std = gaussian_smoothing(y_hot, A=0.5)
plt.plot(y_gs, 'g.-')

y_gst_a, std = gaussian_smoothing(y_hot, A=0.5, mu=5.5)
plt.plot(y_gst_a, 'y.-')

y_gst_b, std = gaussian_smoothing(y_hot, A=0.5, mu=5.8)
plt.plot(y_gst_b, 'c.-')

plt.legend([
    'y_hot', 'label smoothing' + '\n' + '(alpha=0.3)',
    'gaussian smoothing' + '\n' + 'for interval of interest' + '\n' + 'mu=5',
    'gaussian smoothing' + '\n' + 'mu=5.5', 'gaussian smoothing' + '\n' + 'mu=5.8'
])

plt.grid()
plt.xticks(np.arange(11), np.arange(0, 110, 10))
plt.xlabel('''Time (ms)
original (quantized) one hot label:
[0,0,0,0,0,1,0,0,0,0,0]
\n
label smooting is defined as:
 y_ls = (1 - α) * y_hot + α / K,
where K is the number of classes, α is the smoothing parameter
\n
gaussian smoothing for the interval (± 10ms) of interest:
y_gs = A * exp(-(i - mu)**2 / (2 * sigma**2))
with sigma = 0.865 an mu = 5
\n 
gaussian smoothing with unqunatized target timing:
mu = 5.5 for 55ms target timing
''')
