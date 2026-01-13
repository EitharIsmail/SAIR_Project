import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

class LeanringRateS:
    def __init__(self, lr, scheduler_type, decay_rate=0.95, step_size=100):
        self.initial_lr = lr
        self.current_lr = lr
        self.scheduler_type = scheduler_type
        self.decay_rate = decay_rate
        self.step_size = step_size

    def learning_rate_step(self, epoch):
        
        if self.scheduler_type == 'exponential':
            # Formula: lr = initial_lr * (decay_rate ^ epoch)
            self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
            
        elif self.scheduler_type == 'step':
            # Formula: lr drops by decay_rate every 'step_size' epochs
            self.current_lr = self.initial_lr * (self.decay_rate ** (epoch // self.step_size))
            
        else:
            # Default to constant learning rate if type is unknown
            self.current_lr = self.initial_lr
            
        return self.current_lr 
    