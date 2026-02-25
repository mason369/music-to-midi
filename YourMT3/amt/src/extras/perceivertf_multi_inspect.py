import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from matplotlib.animation import FuncAnimation

def l2_normalize(matrix):
    """
    L2 Normalize the matrix along its rows.

    Parameters:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The L2 normalized matrix.
    """
    l2_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / l2_norms
    return normalized_matrix


def z_normalize(matrix):
    """
    Z-normalize the matrix along its rows (mean=0 and std=1).
    Z-normalization is also known as "standardization", and derives from z-score.
    Z = (X - mean) / std
    Z-nomarlized, each row has mean=0 and std=1. 

    Parameters:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The Z normalized matrix.
    """
    mean = np.mean(matrix, axis=1, keepdims=True)
    std = np.std(matrix, axis=1, keepdims=True)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix


def l2_normalize_tensors(tensor_tuple):
    """
    Applies L2 normalization on the last two dimensions for each tensor in a tuple.

    Parameters:
        tensor_tuple (tuple of torch.Tensor): A tuple containing N tensors, each of shape (1, k, 30, 30).

    Returns:
        tuple of torch.Tensor: A tuple containing N L2-normalized tensors.
    """
    normalized_tensors = []
    for tensor in tensor_tuple:
        # Ensure the tensor is a floating-point type
        tensor = tensor.float()

        # Calculate L2 norm on the last two dimensions, keeping the dimensions using keepdim=True
        l2_norm = torch.linalg.norm(tensor, dim=(-2, -1), keepdim=True)

        # Apply L2 normalization
        normalized_tensor = tensor / (
            l2_norm + 1e-7)  # Small value to avoid division by zero

        normalized_tensors.append(normalized_tensor)

    return tuple(normalized_tensors)


def z_normalize_tensors(tensor_tuple):
    """
    Applies Z-normalization on the last two dimensions for each tensor in a tuple.

    Parameters:
        tensor_tuple (tuple of torch.Tensor): A tuple containing N tensors, each of shape (1, k, 30, 30).

    Returns:
        tuple of torch.Tensor: A tuple containing N Z-normalized tensors.
    """
    normalized_tensors = []
    for tensor in tensor_tuple:
        # Ensure the tensor is a floating-point type
        tensor = tensor.float()

        # Calculate mean and std on the last two dimensions
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        std = tensor.std(dim=(-2, -1), keepdim=True)

        # Apply Z-normalization
        normalized_tensor = (tensor - mean) / (
            std + 1e-7)  # Small value to avoid division by zero

        normalized_tensors.append(normalized_tensor)

    return tuple(normalized_tensors)


def apply_temperature_to_attention_tensors(tensor_tuple, temperature=1.0):
    """
    Applies temperature scaling to the attention weights in each tensor in a tuple.
    
    Parameters:
        tensor_tuple (tuple of torch.Tensor): A tuple containing N tensors, 
                                             each of shape (1, k, 30, 30).
        temperature (float): Temperature parameter to control the sharpness 
                             of the attention weights. Default is 1.0.
                             
    Returns:
        tuple of torch.Tensor: A tuple containing N tensors with scaled attention weights.
    """
    scaled_attention_tensors = []

    for tensor in tensor_tuple:
        # Ensure the tensor is a floating-point type
        tensor = tensor.float()

        # Flatten the last two dimensions
        flattened_tensor = tensor.reshape(1, tensor.shape[1],
                                          -1)  # Modified line here

        # Apply temperature scaling and softmax along the last dimension
        scaled_attention = flattened_tensor / temperature
        scaled_attention = F.softmax(scaled_attention, dim=-1)

        # Reshape to original shape
        scaled_attention = scaled_attention.view_as(tensor)

        scaled_attention_tensors.append(scaled_attention)

    return tuple(scaled_attention_tensors)


def shorten_att(tensor_tuple, length=30):
    shortend_tensors = []
    for tensor in tensor_tuple:
        shortend_tensors.append(tensor[:, :, :length, :length])
    return tuple(shortend_tensors)


def keep_top_k(matrix, k=6):
    """
    Keep only the top k values in each row, set the rest to 0.

    Parameters:
        matrix (numpy.ndarray): The input matrix.
        k (int): The number of top values to keep in each row.

    Returns:
        numpy.ndarray: The transformed matrix.
    """
    topk_indices_per_row = np.argpartition(matrix, -k, axis=1)[:, -k:]
    result_matrix = np.zeros_like(matrix)

    for i in range(matrix.shape[0]):
        result_matrix[i, topk_indices_per_row[i]] = matrix[
            i, topk_indices_per_row[i]]
    return result_matrix


def test_case_forward_enc_perceiver_tf_dec_multi_t5():
    import torch
    from model.ymt3 import YourMT3
    from config.config import audio_cfg, model_cfg, shared_cfg
    model_cfg["encoder_type"] = "perceiver-tf"

    model_cfg["encoder"]["perceiver-tf"]["attention_to_channel"] = True
    model_cfg["encoder"]["perceiver-tf"]["num_latents"] = 26

    model_cfg["decoder_type"] = "multi-t5"

    audio_cfg["codec"] = "spec"
    audio_cfg["hop_length"] = 300
    model = YourMT3(audio_cfg=audio_cfg, model_cfg=model_cfg)
    model.eval()

    # x = torch.randn(2, 1, 32767)
    # labels = torch.randint(0, 400, (2, 1024), requires_grad=False)

    # # forward
    # output = model.forward(x, labels)

    # # inference
    # result = model.inference(x, None)

    # display latents
    checkpoint = torch.load(
        "../logs/ymt3/ptf_mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k/checkpoints/model.ckpt",
        map_location="cpu")
    state_dict = checkpoint['state_dict']
    new_state_dict = {
        k: v
        for k, v in state_dict.items() if 'pitchshift' not in k
    }
    model.load_state_dict(new_state_dict, strict=False)

    latents = model.encoder.latent_array.latents.detach().numpy()
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    cos = cosine_similarity(latents)

    from utils.data_modules import AMTDataModule
    from einops import rearrange
    # dm = AMTDataModule(data_preset_multi={"presets": ["slakh"]})
    #dm.setup("test")
    # dl = dm.test_dataloader()
    # ds = list(dl.values())[0].dataset
    # audio, notes, tokens, _ = ds.__getitem__(7)
    # x = audio[[16], ::]
    # label = tokens[[16], :]

    # from utils.task_manager import TaskManager
    # tm = TaskManager(task_name='mc13_256')
    # dm = AMTDataModule(data_preset_multi={"presets": ["slakh"]},
    #                    task_manager=tm,
    #                    train_stem_iaug_prob=None,
    #                    train_stem_xaug_policy=None)
    # dm.setup('fit')
    # dl = dm.train_dataloader()
    # ds = dl.flattened[0].dataset
    # audio,tokens, _, _ = ds.__getitem__(67)
    # x = audio[[5], ::]
    # label = tokens[[5], :]
    # save audio
    # torchaudio.save("singing.wav", x[0, :, :], 16000)
    
    x, _ = torchaudio.load('piano.wav')#'test.wav')
    x = x.unsqueeze(0)

    # spectrogram
    x_spec = model.spectrogram(x)
    x_conv = model.pre_encoder(x_spec)
    # Create a larger figure
    plt.figure(
        figsize=(15,
                 10))  # Adjust these numbers as needed for width and height
    plt.subplot(2, 4, 1)
    plt.imshow(x_spec[0].detach().numpy().T, aspect='auto', origin='lower')
    plt.title("spectrogram")
    plt.xlabel('time step')
    plt.ylabel('frequency bin')
    plt.subplot(2, 4, 2)
    plt.imshow(x_conv[0][:, :, 0].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("conv(spec), ch=0")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.subplot(2, 4, 3)
    plt.imshow(x_conv[0][:, :, 42].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("ch=42")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.subplot(2, 4, 4)
    plt.imshow(x_conv[0][:, :, 80].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("ch=80")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.subplot(2, 4, 5)
    plt.imshow(x_conv[0][:, :, 11].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("ch=11")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.subplot(2, 4, 6)
    plt.imshow(x_conv[0][:, :, 20].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("ch=20")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.subplot(2, 4, 7)
    plt.imshow(x_conv[0][:, :, 77].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("ch=77")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.subplot(2, 4, 8)
    plt.imshow(x_conv[0][:, :, 90].detach().numpy().T,
               aspect='auto',
               origin='lower')
    plt.title("ch=90")
    plt.xlabel('time step')
    plt.ylabel('F')
    plt.tight_layout()
    plt.show()

    # encoding
    output = model.encoder(inputs_embeds=x_conv,
                           output_hidden_states=True,
                           output_attentions=True)
    enc_hs_all, att, catt = output["hidden_states"], output[
        "attentions"], output["cross_attentions"]
    enc_hs_last = enc_hs_all[2]

    # enc_hs: time-varying encoder hidden state
    plt.subplot(2, 3, 1)
    plt.imshow(enc_hs_all[0][0][:, :, 21].detach().numpy().T)
    plt.title('ENC_HS B0, d21')
    plt.colorbar(orientation='horizontal')
    plt.ylabel('latent k')
    plt.xlabel('t')
    plt.subplot(2, 3, 4)
    plt.imshow(enc_hs_all[0][0][:, :, 127].detach().numpy().T)
    plt.colorbar(orientation='horizontal')
    plt.title('B0, d127')
    plt.ylabel('latent k')
    plt.xlabel('t')
    plt.subplot(2, 3, 2)
    plt.imshow(enc_hs_all[1][0][:, :, 21].detach().numpy().T)
    plt.colorbar(orientation='horizontal')
    plt.title('B1, d21')
    plt.ylabel('latent k')
    plt.xlabel('t')
    plt.subplot(2, 3, 5)
    plt.imshow(enc_hs_all[1][0][:, :, 127].detach().numpy().T)
    plt.colorbar(orientation='horizontal')
    plt.title('B1, d127')
    plt.ylabel('latent k')
    plt.xlabel('t')
    plt.subplot(2, 3, 3)
    plt.imshow(enc_hs_all[2][0][:, :, 21].detach().numpy().T)
    plt.colorbar(orientation='horizontal')
    plt.title('B2, d21')
    plt.ylabel('latent k')
    plt.xlabel('t')
    plt.subplot(2, 3, 6)
    plt.imshow(enc_hs_all[2][0][:, :, 127].detach().numpy().T)
    plt.colorbar(orientation='horizontal')
    plt.title('B2, d127')
    plt.ylabel('latent k')
    plt.xlabel('t')
    plt.tight_layout()
    plt.show()

    # enc_hs: time-varying encoder hidden state by k (block, 1, t, k, d)
    # --> (t, d) for each k in last block
    data = enc_hs_all[2][0].detach().numpy()  # (T, K, D)
    fig, axs = plt.subplots(
        5, 5, figsize=(10, 9))  # 25 subplots arranged in 5 rows and 5 columns
    axs = axs.flatten(
    )  # Flatten the 2D array of axes to 1D for easy iteration

    for k in range(25):  # Iterating through K indices from 0 to 24
        axs[k].imshow(data[:, k, :].T,
                      cmap='viridis')  # Transposing the matrix to swap T and D
        axs[k].set_title(f'k={k}')
        axs[k].set_xlabel('Time step')
        axs[k].set_ylabel('Dim')

    # Adjusting layout for better visibility
    plt.tight_layout()
    plt.show()

    #!! Projected encoder hidden state for 13 channels, that is conditioning for decoder
    enc_hs_proj = model.pre_decoder(enc_hs_last)
    fig, axs = plt.subplots(1, 13, figsize=(26, 8))  # 13 subplots in a row
    data = enc_hs_proj[0].detach().numpy()
    for ch in range(13):
        axs[ch].imshow(np.rot90(data[ch]), cmap='viridis')  # Rotate 90 degrees
        axs[ch].set_title(f'ch: {ch}')
        axs[ch].set_xlabel('Time step')
        axs[ch].set_ylabel('Dim')
    plt.suptitle(
        'linear projection of encoder outputs by channel, which is conditioning for enc-dec cross attention',
        y=0.1,
        fontsize=12)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

    plt.subplot(221)
    plt.imshow(enc_hs_all[2][0][0, :, :].detach().numpy(), aspect='auto')
    plt.title('enc_hs, t=0')
    plt.ylabel('latent k')
    plt.xlabel('d')
    plt.subplot(222)
    plt.imshow(enc_hs_all[2][0][10, :, :].detach().numpy(), aspect='auto')
    plt.title('enc_hs, t=10')
    plt.ylabel('latent k')
    plt.xlabel('d')
    plt.subplot(223)
    plt.imshow(enc_hs_all[2][0][20, :, :].detach().numpy(), aspect='auto')
    plt.title('enc_hs, t=20')
    plt.ylabel('latent k')
    plt.xlabel('d')
    plt.subplot(224)
    plt.imshow(enc_hs_all[2][0][30, :, :].detach().numpy(), aspect='auto')
    plt.title('enc_hs, t=30')
    plt.ylabel('latent k')
    plt.xlabel('d')
    plt.tight_layout()
    plt.show()

    # enc_hs correlation: which dim has most unique info?
    plt.subplot(1, 3, 1)
    a = rearrange(enc_hs_last, '1 t k d -> t (k d)').detach().numpy()
    plt.imshow(cosine_similarity(a))
    plt.title("enc hs, t x t cos_sim")
    plt.subplot(1, 3, 2)
    b = rearrange(enc_hs_last, '1 t k d -> k (t d)').detach().numpy()
    plt.imshow(cosine_similarity(b))
    plt.title("enc hs, k x k cos_sim")
    plt.subplot(1, 3, 3)
    c = rearrange(enc_hs_last, '1 t k d -> d (k t)').detach().numpy()
    plt.imshow(cosine_similarity(c))
    plt.title("cross att, d x d cos_sim")
    plt.tight_layout()
    plt.show()

    #!! enc latent
    plt.imshow(model.encoder.latent_array.latents.detach().numpy())
    plt.title('latent array')
    plt.xlabel('d')
    plt.ylabel('latent k')
    plt.show()

    #!! enc Spectral Cross Attention: (T x head x K x D). How latent K attends to conv channel C?
    plt.subplot(311)
    plt.imshow(
        torch.sum(torch.sum(catt[0][0], axis=0), axis=0).detach().numpy())
    plt.title('block=0')
    plt.ylabel('latent k')
    plt.xlabel('conv channel')
    plt.subplot(312)
    plt.imshow(
        torch.sum(torch.sum(catt[1][0], axis=0), axis=0).detach().numpy())
    plt.title('block=1')
    plt.ylabel('latent k')
    plt.xlabel('conv channel')
    plt.subplot(313)
    plt.imshow(
        torch.sum(torch.sum(catt[2][0], axis=0), axis=0).detach().numpy())
    plt.title('block=2')
    plt.ylabel('latent k')
    plt.xlabel('conv channel')
    # f'spectral cross attention. T-C-F Model',
    # y=0,
    # fontsize=12)
    plt.tight_layout()
    plt.show()

    #!! Animation of SCA for varying time, head in last block
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # Adjusted figsize for better layout

    # Function to update the plots for each frame in the animation
    def update(t):
        # Clear previous images
        ax1.clear()
        ax2.clear()

        # Update subplot for h=3
        ax1.imshow(catt[2][0][t, 3, :, :].detach().numpy())
        ax1.set_title(f'block=2, t={t}, head=3')
        ax1.set_ylabel('latent k'); ax1.set_xlabel('conv channel')

        # Update subplot for h=5
        ax2.imshow(catt[2][0][t, 5, :, :].detach().numpy())
        ax2.set_title(f'block=2, t={t}, head=5')
        ax2.set_ylabel('latent k'); ax2.set_xlabel('conv channel')

        # Adjust layout
        fig.tight_layout()

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, 110), interval=200)
    anim.save('animation.gif', writer='pillow', fps=5)



    fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 1, 0.5]})  # Adjusted for different subplot sizes

    # Subplots for catt visualization (h=3 and h=5)
    ax_catt3, ax_catt5, ax_att_row = axs

    # Creating 8 subplots for att visualization within the third row
    for i in range(8):
        ax_att_row = fig.add_subplot(3, 8, 17 + i)  # Adding subplots in the third row

    # Update function for the combined animation
    def combined_update_smaller_att(t):
        # Update subplot for catt with h=3
        ax_catt3.clear()
        ax_catt3.imshow(catt[2][0][t, 3, :, :].detach().numpy())
        ax_catt3.set_title(f'block=2, t={t}, head=3')
        ax_catt3.set_ylabel('latent k'); ax_catt3.set_xlabel('conv channel')

        # Update subplot for catt with h=5
        ax_catt5.clear()
        ax_catt5.imshow(catt[2][0][t, 5, :, :].detach().numpy())
        ax_catt5.set_title(f'block=2, t={t}, head=5')
        ax_catt5.set_ylabel('latent k'); ax_catt5.set_xlabel('conv channel')

        # Update subplots for att (8 heads in one row)
        for i in range(8):
            ax = fig.add_subplot(3, 8, 17 + i)
            ax.clear()
            ax.imshow(att[0][1][t, i, :, :].detach().numpy(), cmap='viridis')
            ax.set_title(f't={t}, head={i}')
            ax.set_xlabel('k')
            ax.set_ylabel('k')
            ax.axis('square')  # Make each subplot square-shaped

        # Adjust layout
        fig.tight_layout()
    combined_anim_smaller_att = FuncAnimation(fig, combined_update_smaller_att, frames=range(0, 110), interval=200)
    combined_anim_smaller_att.save('combined_animation_smaller_att.gif', writer='pillow', fps=5)





    # enc Latent Self-attention: How latent K attends to K?
    plt.subplot(231)
    plt.imshow(torch.sum(torch.sum(att[0][0], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B0L0')
    plt.xlabel('latent k')
    plt.ylabel('latent k')
    plt.subplot(234)
    plt.imshow(torch.sum(torch.sum(att[0][1], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B0L1')
    plt.xlabel('latent k')
    plt.ylabel('latent k')
    plt.subplot(232)
    plt.imshow(torch.sum(torch.sum(att[1][0], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B1L0')
    plt.xlabel('latent k')
    plt.ylabel('latent k')
    plt.subplot(235)
    plt.imshow(torch.sum(torch.sum(att[1][1], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B1L1')
    plt.xlabel('latent k')
    plt.ylabel('latent k')
    plt.subplot(233)
    plt.imshow(torch.sum(torch.sum(att[2][0], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B2L0')
    plt.xlabel('latent k')
    plt.ylabel('latent k')
    plt.subplot(236)
    plt.imshow(torch.sum(torch.sum(att[2][1], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B2L1')
    plt.xlabel('latent k')
    plt.ylabel('latent k')
    plt.tight_layout()
    plt.show()
    # Time varying, different head for latent self-attention
    #!!! Display latent self-attention for each head
    bl = 0  # first latent transformer block, last layer att
    data = att[bl][1].detach().numpy()
    time_steps = [30, 50, 100]
    fig, axs = plt.subplots(
        len(time_steps), 8,
        figsize=(16, 6))  # Subplots for each time step and head
    for i, t in enumerate(time_steps):
        for head in range(8):
            axs[i, head].imshow(data[t, head, :, :], cmap='viridis')
            axs[i, head].set_title(f't={t}, head={head}')
            axs[i, head].set_xlabel('k')
            axs[i, head].set_ylabel('k')
    plt.suptitle(
        f'latent transformer block={bl}, last layer self-attention over time',
        y=0,
        fontsize=12)
    plt.tight_layout()
    plt.show()

    bl = 1  # second latent transformer block, last layer att
    data = att[bl][1].detach().numpy()
    time_steps = [30, 50, 100]
    fig, axs = plt.subplots(
        len(time_steps), 8,
        figsize=(16, 6))  # Subplots for each time step and head
    for i, t in enumerate(time_steps):
        for head in range(8):
            axs[i, head].imshow(data[t, head, :, :], cmap='viridis')
            axs[i, head].set_title(f't={t}, head={head}')
            axs[i, head].set_xlabel('k')
            axs[i, head].set_ylabel('k')
    plt.suptitle(
        f'latent transformer block={bl}, last layer self-attention over time',
        y=0,
        fontsize=12)
    plt.tight_layout()
    plt.show()

    bl = 2  # last latent transformer block, last layer att
    data = att[bl][1].detach().numpy()
    time_steps = [30, 50, 100]
    fig, axs = plt.subplots(
        len(time_steps), 8,
        figsize=(16, 6))  # Subplots for each time step and head
    for i, t in enumerate(time_steps):
        for head in range(8):
            axs[i, head].imshow(data[t, head, :, :], cmap='viridis')
            axs[i, head].set_title(f't={t}, head={head}')
            axs[i, head].set_xlabel('k')
            axs[i, head].set_ylabel('k')
    plt.suptitle(
        f'latent transformer block={bl}, last layer self-attention over time',
        y=0,
        fontsize=12)
    plt.tight_layout()
    plt.show()

    # Temporal Self-attention: (K x H x T x T) How time t attends to time t?
    plt.subplot(231)
    plt.imshow(torch.sum(torch.sum(att[0][2], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B0L2')
    plt.xlabel('t')
    plt.ylabel('t')
    plt.subplot(234)
    plt.imshow(torch.sum(torch.sum(att[0][3], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B0L3')
    plt.xlabel('t')
    plt.ylabel('t')
    plt.subplot(232)
    plt.imshow(torch.sum(torch.sum(att[1][2], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B1L2')
    plt.xlabel('t')
    plt.ylabel('t')
    plt.subplot(235)
    plt.imshow(torch.sum(torch.sum(att[1][3], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B1L3')
    plt.xlabel('t')
    plt.ylabel('t')
    plt.subplot(233)
    plt.imshow(torch.sum(torch.sum(att[2][2], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B2L2')
    plt.xlabel('t')
    plt.ylabel('t')
    plt.subplot(236)
    plt.imshow(torch.sum(torch.sum(att[2][3], axis=1),
                         axis=0).detach().numpy(),
               origin='upper')
    plt.title('B2L3')
    plt.xlabel('t')
    plt.ylabel('t')
    plt.tight_layout()
    plt.show()

    # decoding
    dec_input_ids = model.shift_right_fn(label)
    dec_inputs_embeds = model.embed_tokens(dec_input_ids)
    dec_output = model.decoder(inputs_embeds=dec_inputs_embeds,
                               encoder_hidden_states=enc_hs_proj,
                               output_attentions=True,
                               output_hidden_states=True,
                               return_dict=True)
    dec_att, dec_catt = dec_output.attentions, dec_output.cross_attentions
    dec_hs_all = dec_output.hidden_states
    dec_last_hs = dec_output.last_hidden_state

    # lm head
    logits = model.lm_head(dec_last_hs)

    # pred ids
    pred_ids = torch.argmax(logits, dim=3)

    # dec att
    plt.subplot(1, 2, 1)
    plt.imshow(torch.sum(dec_att[5][0], axis=0).detach().numpy())
    plt.title('decoder attention, layer0')
    plt.xlabel('decoder time step')
    plt.ylabel('decoder time step')
    plt.subplot(1, 2, 2)
    plt.imshow(torch.sum(dec_att[7][0], axis=0).detach().numpy())
    plt.title('decoder attention, final layer')
    plt.xlabel('decoder step')
    plt.show()
    
    
    # dec catt
    def remove_values_after_eos(catt_np, pred_ids, max_k):
        # catt_np: (k, head, t, t)
        # pred_ids: (1, k, t))
        max_length = pred_ids.shape[-1]
        seq_lengths = np.zeros((max_k), dtype=np.int32)
        for k in range(max_k):
            for t in range(max_length):
                if pred_ids[0, k, t] == 1:
                    break
            catt_np[k, :, t+1:, :] = 0
            # catt_np[k, :, :, t+1:] = 0
            seq_lengths[k] = t+1  
        return catt_np, seq_lengths

    # data = dec_catt[1].detach().numpy() # last layer's cross attention
    l = 4
    data = dec_catt[l].detach().numpy() 
    data, seq_lengths = remove_values_after_eos(data, pred_ids, max_k=13)
    seq_lengths[:]= 256

    fig, axs = plt.subplots(13, 6, figsize=(21, 39))  # 13 rows (for k=0:12) and 7 columns (for head=0:6)
    for k in range(13):
        s = seq_lengths[k]
        for head in range(6):
            axs[k, head].imshow(data[k, head, :s, :].T, aspect='auto', cmap='viridis')
            axs[k, head].set_title(f'Layer {l}, k={k}, head={head}')
            axs[k, head].set_xlabel('Decoder step')
            axs[k, head].set_ylabel('Encoder frame')
    plt.tight_layout()
    plt.show()


    # # dec catt by head with xxx
    # dec_att_z = z_normalize_tensors(shorten_att(dec_att))
    # plt.imshow(dec_att_z[0][0, 0, :, :].detach().numpy())
    # from bertviz import head_view
    # token = []
    # for i in label[0, :30]:
    #     token.append(str(i))
    # head_view(dec_att_z, tokens)

    # dec_hs
    plt.subplot(1, 2, 1)
    k=2
    plt.imshow(dec_last_hs[0][k].detach().numpy(), origin='upper')
    plt.colorbar(orientation='horizontal')
    plt.title('decoder last hidden state, k=0')
    plt.xlabel('hidden dim')
    plt.ylabel('time step')
    plt.subplot(1, 2, 2)
    k=12
    plt.imshow(dec_last_hs[0][k].detach().numpy(), origin='upper')
    plt.colorbar(orientation='horizontal')
    plt.title('decoder last hidden state, k=12')
    plt.xlabel('hidden dim')
    plt.show()

    # lm head
    logits = model.lm_head(dec_last_hs)
    k=6
    plt.imshow(logits[0][k][0:200, :].detach().numpy().T, origin='upper')
    plt.title('lm head output')
    plt.xlabel('vocab dim')
    plt.ylabel('time step')
    plt.show()
    softmax = torch.nn.Softmax(dim=3)
    logits_sm = softmax(logits) # B, K, T, V
    k=6
    plt.imshow(logits_sm[0][k][:255, :].detach().numpy().T, origin='upper')
    plt.title('lm head softmax')
    plt.xlabel('vocab dim')
    plt.ylabel('time step')
    # plt.xlim([1000, 1350])
    plt.show()

    k = 10
    print(torch.argmax(logits, dim=3)[0,k,:])




