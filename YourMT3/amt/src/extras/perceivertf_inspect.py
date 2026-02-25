import numpy as np
import torch
import torch.nn.functional as F


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


def test_case_forward_enc_perceiver_tf_dec_t5():
    import torch
    from model.ymt3 import YourMT3
    from config.config import audio_cfg, model_cfg, shared_cfg
    model_cfg["encoder_type"] = "perceiver-tf"
    model_cfg["encoder"]["perceiver-tf"]["attention_to_channel"] = True
    model_cfg["encoder"]["perceiver-tf"]["num_latents"] = 24
    model_cfg["decoder_type"] = "t5"
    model_cfg["pre_decoder_type"] = "default"

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
        "../logs/ymt3/ptf_all_cross_rebal5_spec300_xk2_amp0811_edr_005_attend_c_full_plus_b52/checkpoints/model.ckpt",
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
    dm = AMTDataModule(data_preset_multi={"presets": ["slakh"]})
    dm.setup("test")
    dl = dm.test_dataloader()
    ds = list(dl.values())[0].dataset
    audio, notes, tokens, _ = ds.__getitem__(7)
    x = audio[[16], ::]
    label = tokens[[16], :]
    # spectrogram
    x_spec = model.spectrogram(x)
    plt.imshow(x_spec[0].detach().numpy().T, aspect='auto', origin='lower')
    plt.title("spectrogram")
    plt.xlabel('time step')
    plt.ylabel('frequency bin')
    plt.show()
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

    enc_hs_proj = model.pre_decoder(enc_hs_last)
    plt.imshow(enc_hs_proj[0].detach().numpy())
    plt.title(
        'ENC_HS_PROJ: linear projection of encoder output, which is used for enc-dec cross attention'
    )
    plt.colorbar(orientation='horizontal')
    plt.ylabel('latent k')
    plt.xlabel('d')
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

    # enc latent
    plt.imshow(model.encoder.latent_array.latents.detach().numpy())
    plt.title('latent array')
    plt.xlabel('d')
    plt.ylabel('latent k')
    plt.show()

    # enc Spectral Cross Attention: (T x head x K x D). How latent K attends to conv channel C?
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
    plt.tight_layout()
    plt.show()
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
    plt.subplot(231)
    plt.imshow(att[0][0][30, 3, :, :].detach().numpy())
    plt.title('B0L0, t=30, Head=3')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(234)
    plt.imshow(att[0][1][30, 3, :, :].detach().numpy())
    plt.title('B0L1, t=30, Head=3')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(232)
    plt.imshow(att[1][0][30, 3, :, :].detach().numpy())
    plt.title('B1L0, t=30, Head=3')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(235)
    plt.imshow(att[1][1][30, 3, :, :].detach().numpy())
    plt.title('B1L1, t=30, Head=3')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(233)
    plt.imshow(att[2][0][30, 3, :, :].detach().numpy())
    plt.title('B2L0, t=30, Head=3')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(236)
    plt.imshow(att[2][1][30, 3, :, :].detach().numpy())
    plt.title('B2L1, t=30, Head=3')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.tight_layout()
    plt.show()
    plt.subplot(231)
    plt.imshow(att[0][0][30, 5, :, :].detach().numpy())
    plt.title('B0L0, t=30, Head=5')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(234)
    plt.imshow(att[0][1][30, 5, :, :].detach().numpy())
    plt.title('B0L1, t=30, Head=5')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(232)
    plt.imshow(att[1][0][30, 5, :, :].detach().numpy())
    plt.title('B1L0, t=30, Head=5')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(235)
    plt.imshow(att[1][1][30, 5, :, :].detach().numpy())
    plt.title('B1L1, t=30, Head=5')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(233)
    plt.imshow(att[2][0][30, 5, :, :].detach().numpy())
    plt.title('B2L0, t=30, Head=5')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
    plt.subplot(236)
    plt.imshow(att[2][1][30, 5, :, :].detach().numpy())
    plt.title('B2L1, t=30, Head=5')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('k')
    plt.ylabel('k')
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

    # dec att
    plt.subplot(1, 2, 1)
    plt.imshow(torch.sum(dec_att[0][0], axis=0).detach().numpy())
    plt.title('decoder attention, layer0')
    plt.xlabel('decoder time step')
    plt.ylabel('decoder time step')
    plt.subplot(1, 2, 2)
    plt.imshow(torch.sum(dec_att[7][0], axis=0).detach().numpy())
    plt.title('decoder attention, layer8')
    plt.xlabel('decoder time step')
    plt.show()
    # dec catt
    plt.imshow(np.rot90((torch.sum(dec_catt[7][0],
                                   axis=0))[:1000, :].detach().numpy()),
               origin='upper',
               aspect='auto')
    plt.colorbar()
    plt.title('decoder cross att, layer8')
    plt.xlabel('decoder time step')
    plt.ylabel('encoder frame')
    plt.show()
    # dec catt by head with xxx
    dec_att_z = z_normalize_tensors(shorten_att(dec_att))
    plt.imshow(dec_att_z[0][0, 0, :, :].detach().numpy())
    from bertviz import head_view
    token = []
    for i in label[0, :30]:
        token.append(str(i))
    head_view(dec_att_z, tokens)

    # dec_hs
    plt.subplot(1, 2, 1)
    plt.imshow(dec_hs_all[0][0].detach().numpy(), origin='upper')
    plt.colorbar(orientation='horizontal')
    plt.title('decoder hidden state, layer1')
    plt.xlabel('hidden dim')
    plt.ylabel('time step')
    plt.subplot(1, 2, 2)
    plt.imshow(dec_hs_all[7][0].detach().numpy(), origin='upper')
    plt.colorbar(orientation='horizontal')
    plt.title('decoder hidden state, layer8')
    plt.xlabel('hidden dim')
    plt.show()

    # lm head
    logits = model.lm_head(dec_hs_all[0])
    plt.imshow(logits[0][0:200, :].detach().numpy(), origin='upper')
    plt.title('lm head softmax')
    plt.xlabel('vocab dim')
    plt.ylabel('time step')
    plt.xlim([1000, 1350])
    plt.show()
    softmax = torch.nn.Softmax(dim=2)
    logits_sm = softmax(logits)
    plt.imshow(logits_sm[0][0:200, :].detach().numpy(), origin='upper')
    plt.title('lm head softmax')
    plt.xlabel('vocab dim')
    plt.ylabel('time step')
    plt.xlim([1000, 1350])
    plt.show()
