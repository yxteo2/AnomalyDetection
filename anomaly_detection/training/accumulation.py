"""Sample-weighted accumulation without retaining microbatch graphs."""

import torch


class GradientAccumulator:
    def __init__(self, optimizer, batches=1, max_norm=None):
        if type(batches) is not int or batches < 1:
            raise ValueError("accumulate_grad_batches must be a positive integer.")
        self.optimizer, self.batches, self.max_norm = optimizer, batches, max_norm
        self.parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        self.samples = self.microbatches = 0
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, sample_count):
        (loss * sample_count).backward()
        self.samples += sample_count
        self.microbatches += 1
        if self.microbatches == self.batches:
            self.step()

    def step(self):
        """Also flush a final partial accumulation window at the end of an epoch."""
        if not self.samples:
            return
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(self.samples)
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.samples = self.microbatches = 0
