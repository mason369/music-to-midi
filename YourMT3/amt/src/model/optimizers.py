""" optimizers.py 

Code based on nanoT5 project:
    https://github.com/PiotrNawrot/nanoT5/blob/main/nanoT5/utils/copied_utils.py

+ D-adapt Adam from https://github.com/facebookresearch/dadaptation
"""
import importlib
import math
import torch

from typing import Iterable, Tuple
from torch import nn
from torch.optim import Optimizer
from transformers import Adafactor
from torch.optim import AdamW


class AdamWScale(Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel()**0.5)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1**state["step"]
                    bias_correction2 = 1.0 - beta2**state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # /Adapt Step from Adagrad
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adagrad

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


# def get_optimizer(models_dict: nn.ModuleDict,
#                   optimizer_name: str,
#                   base_lr: float,
#                   weight_decay: float = 0.):

#     no_decay = [
#         "bias", "LayerNorm", "layernorm", "layer_norm", "ln", "BatchNorm", "bn", "batch_norm",
#         "batchnorm"
#     ]


#     optimizer_grouped_parameters = []
#     for name, current_model in models_dict.items():
#         if current_model is None:
#             continue
#         optimizer_grouped_parameters += [
#             {
#                 "params": [
#                     p for n, p in current_model.named_parameters()
#                     if not any(nd in n for nd in no_decay)
#                 ],
#                 "weight_decay": weight_decay,
#             },
#             {
#                 "params": [
#                     p for n, p in current_model.named_parameters()
#                     if any(nd in n for nd in no_decay)
#                 ],
#                 "weight_decay": 0.0,
#             },
#         ]
def get_optimizer(models_dict: nn.ModuleDict,
                  optimizer_name: str,
                  base_lr: float,
                  weight_decay: float = 0.):

    no_decay = [
        "bias", "LayerNorm", "layernorm", "layer_norm", "ln", "BatchNorm", "bn", "batch_norm",
        "batchnorm"
    ]
    optimizer_grouped_parameters = []
    for n, p in models_dict:
        # drop pitch shifter
        if 'pshifters' in n:
            continue
        # no decay
        if n in no_decay:
            optimizer_grouped_parameters.append({"params": [p], "weight_decay": 0.0})
        else:
            optimizer_grouped_parameters.append({"params": [p], "weight_decay": weight_decay})

    if optimizer_name.lower() == 'adamw':
        base_lr = 1e-03 if base_lr == None else float(base_lr)
        opt = AdamW(optimizer_grouped_parameters, lr=base_lr)
    elif optimizer_name.lower() == 'adafactor':
        if base_lr == None:
            opt = Adafactor(
                optimizer_grouped_parameters,
                lr=None,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True)
        else:
            opt = Adafactor(optimizer_grouped_parameters, lr=base_lr, relative_step=False)
    elif optimizer_name.lower() == 'adamwscale':
        base_lr = 1e-02 if base_lr == None else float(base_lr)
        opt = AdamWScale(
            optimizer_grouped_parameters,
            lr=base_lr,
        )
    elif optimizer_name.lower() == 'cpuadam':
        dspd = importlib.import_module('deepspeed')
        base_lr = 1e-03 if base_lr == None else float(base_lr)
        opt = dspd.ops.adam.cpu_adam.DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=base_lr)
    elif optimizer_name.lower() == 'dadaptadam':
        dadaptation = importlib.import_module('dadaptation')
        base_lr = 1.0 if base_lr == None else float(base_lr)
        opt = dadaptation.DAdaptAdam(optimizer_grouped_parameters, lr=base_lr)
    else:
        raise NotImplementedError(optimizer_name)

    return opt, base_lr
