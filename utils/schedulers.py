import math

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


class LinearRampupRampdownScheduler:
    """
    Linear ramp-up from init_lr to max_lr by rampup_steps then 
    linear ramp-down to min_lr by rampdown_steps
    """
    def __init__(self, optimizer, init_lr, max_lr, min_lr, rampup_steps, rampdown_steps):
        self.optimizer = optimizer
        self.step_counter = 0
        # Linear rampup
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.rampup_steps = rampup_steps
        self.incr_per_step = float(max_lr - init_lr) / rampup_steps
        # Linear rampdown
        self.min_lr = min_lr
        self.rampdown_steps = rampdown_steps
        self.decr_per_step = float(min_lr - max_lr) / rampdown_steps
    
    def step(self):
        if self.step_counter < self.rampup_steps:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr'] + self.incr_per_step
        elif self.step_counter < self.rampup_steps + self.rampdown_steps:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr'] + self.decr_per_step
        self.step_counter += 1
        

class LinearRampupCosineAnnealingScheduler:
    """
    """
    def __init__(self, optimizer, init_lr, max_lr, min_lr, rampup_steps, cycle_steps, max_lr_decay=None, rampdown_cycles=None):
        assert (max_lr_decay is not None or rampdown_cycles is not None) and not (max_lr_decay and rampdown_cycles), "Requires either decay or rampdown parameter"
        self.optimizer = optimizer
        self.step_counter = 0
        self.cycle_counter = 0
        # Linear rampup
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.rampup_steps = rampup_steps
        self.incr_per_step = float(max_lr - init_lr) / rampup_steps
        # Cosine annealing
        self.min_lr = min_lr
        self.cycle_steps = cycle_steps
        # Maximum Decrement
        if max_lr_decay is not None:
            self.get_next_max_lr = lambda old_max_lr: ((old_max_lr - self.min_lr) * max_lr_decay) + self.min_lr
        else:
            decr_per_step = float(min_lr - max_lr) / rampdown_cycles
            self.get_next_max_lr = lambda old_max_lr: max(old_max_lr + decr_per_step, self.min_lr)
    
    def step(self):
        if self.step_counter < self.rampup_steps:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr'] + self.incr_per_step
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (self.cycle_counter) / self.cycle_steps))
            if self.cycle_counter == self.cycle_steps - 1:
                self.cycle_counter = 0
                self.max_lr = self.get_next_max_lr(self.max_lr)
            else:
                self.cycle_counter += 1
        self.step_counter += 1
