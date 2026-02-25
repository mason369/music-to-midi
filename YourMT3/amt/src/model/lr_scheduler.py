# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""lr_schedule.py"""
import torch
from typing import Dict, Optional


def get_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, base_lr: float, scheduler_cfg: Dict):

    if scheduler_name.lower() == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=scheduler_cfg["warmup_steps"],
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg["total_steps"] - scheduler_cfg["warmup_steps"],
            eta_min=scheduler_cfg["final_cosine"],
        )

        lr_scheduler = SequentialLR(optimizer,
                                    schedulers=[scheduler1, scheduler2],
                                    milestones=[scheduler_cfg["warmup_steps"]])
    elif scheduler_name.lower() == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        print(msg)

        num_steps_optimizer1 = math.ceil(scheduler_cfg["total_steps"] * 0.9)
        iters_left_for_optimizer2 = scheduler_cfg["total_steps"] - num_steps_optimizer1

        scheduler1 = LambdaLR(optimizer, lambda step: min(base_lr, 1.0 / math.sqrt(step)) / base_lr
                              if step else base_lr / base_lr)

        scheduler2 = LinearLR(optimizer,
                              start_factor=(min(base_lr, 1.0 / math.sqrt(num_steps_optimizer1)) / base_lr),
                              end_factor=0,
                              total_iters=iters_left_for_optimizer2,
                              last_epoch=-1)

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1],
        )
    elif scheduler_name.lower() == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=scheduler_name.lower(),
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler


def extra_stats(args, model, optimizer):
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = sum(p.detach().norm(2).item()**2 for p in model.parameters())**0.5
        stats['weights_l2'] = weights_l2

    cur_lr = optimizer.param_groups[0]['lr']
    stats['lr'] = cur_lr

    return stats
