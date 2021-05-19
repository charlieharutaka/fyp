import torch

class LinearRampupDecayScheduler:
    """
    Linear ramp-up from init_lr to max_lr by rampup_steps then 
    decay by decay_factor every decay_steps.
    """
    def __init__(self, optimizer, init_lr, max_lr, rampup_steps, decay_factor, decay_steps):
        self.optimizer = optimizer
        self.step_counter = 0
        # Linear rampup
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.rampup_steps = rampup_steps
        self.incr_per_step = float(max_lr - init_lr) / rampup_steps
        # Step decay
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
    
    def step(self):
        if self.step_counter < self.rampup_steps:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr'] + self.incr_per_step
        elif (self.step_counter - self.rampup_steps + 1) % self.decay_steps == 0:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr'] * self.decay_factor
        self.step_counter += 1