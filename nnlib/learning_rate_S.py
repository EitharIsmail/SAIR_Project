class LeanringRateS:
    def __init__(self, lr, scheduler_type):
        self.lr = lr
        self.initial_lr = lr
        self.scheduler_type = scheduler_type

    def learning_rate_step(self):
        