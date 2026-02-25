import numpy as np

a = np.arange(12).reshape(2, 3, 2)  # (batch, channel, dim)
print(a)
array([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])

swap_mat = create_swap_channel_mat(input_shape, swap_channel=(1, 2))

# will swap channel 1 and 2 of batch 0 with channel 1 and 2 of batch 1
b = a @ swap_mat
print(b)
# expected output
array([[[0, 1], [8, 9], [10, 11]], [[6, 7], [2, 3], [4, 5]]])

import torch


def swap_channels_between_batches(a_tensor, swap_channels):
    # Copy the tensor to avoid modifying the original tensor
    result_tensor = a_tensor.clone()

    # Unpack the channels to be swapped
    ch1, ch2 = swap_channels

    # Swap the specified channels between batches
    result_tensor[0, ch1, :], result_tensor[1, ch1, :] = a_tensor[1, ch1, :].clone(), a_tensor[0, ch1, :].clone()
    result_tensor[0, ch2, :], result_tensor[1, ch2, :] = a_tensor[1, ch2, :].clone(), a_tensor[0, ch2, :].clone()

    return result_tensor


# Define a sample tensor 'a_tensor'
a_tensor = torch.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], dtype=torch.float32)

# Define channels to swap
swap_channels = (1, 2)  # Channels to swap between batches

# Swap the channels between batches
swapped_tensor = swap_channels_between_batches(a_tensor, swap_channels)

# Print the original tensor and the tensor after swapping channels between batches
print("Original Tensor 'a_tensor':")
print(a_tensor)
print("\nTensor after swapping channels between batches:")
print(swapped_tensor)

#-------------------------------------------------

import torch
from einops import rearrange


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def create_batch_swap_matrix(batch_size, channels, swap_channels):
    swap_mat = np.eye(batch_size * channels)

    for c in swap_channels:
        idx1 = c  # 첫 번째 배치의 교환할 채널 인덱스
        idx2 = c + channels  # 두 번째 배치의 교환할 채널 인덱스

        swap_mat[idx1, idx1], swap_mat[idx2, idx2] = 0, 0  # 대각선 값을 0으로 설정
        swap_mat[idx1, idx2], swap_mat[idx2, idx1] = 1, 1  # 해당 채널을 교환
    return swap_mat


def create_batch_swap_matrix(batch_size, channels, swap_channels):
    swap_mat = np.eye(batch_size * channels)

    # 모든 채널에 대해 교환 수행
    for c in swap_channels:
        idx1 = np.arange(c, batch_size * channels, channels)  # 현재 채널의 모든 배치 인덱스
        idx2 = (idx1 + channels) % (batch_size * channels)  # 순환을 위해 modulo 사용

        swap_mat[idx1, idx1] = 0
        swap_mat[idx2, idx2] = 0
        swap_mat[idx1, idx2] = 1
        swap_mat[idx2, idx1] = 1

    return swap_mat


def swap_channels_between_batches(input_tensor, swap_matrix):
    reshaped_tensor = rearrange(input_tensor, 'b c d -> (b c) d')
    swapped_tensor = swap_matrix @ reshaped_tensor
    return rearrange(swapped_tensor, '(b c) d -> b c d', b=input_tensor.shape[0])


# 예제 파라미터
batch_size = 2
channels = 3
# swap_info  = {
#     : [1, 2] # batch_index: [channel_indices]
# }
swap_channels = [1, 2]  # 교환할 채널

# 예제 텐서 생성
input_tensor = torch.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], dtype=torch.float32)

# swap matrix 생성
swap_matrix = create_batch_swap_matrix(batch_size, channels, swap_channels)
swap_matrix = torch.Tensor(swap_matrix)

# 채널 교환 수행
swapped_tensor = swap_channels_between_batches(input_tensor, swap_matrix)

# 결과 출력
print("Original Tensor:")
print(input_tensor)
print("\nSwapped Tensor:")
print(swapped_tensor)
