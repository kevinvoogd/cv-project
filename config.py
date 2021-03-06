# Implementation of Swin Tansformer
# in Pytorch.
# Created by: Guru Deep Singh, Kevin Luis Voogd

#Script to define variables used in the repository
class Config():
    def __init__(self):
        self.input_size = 224 # Side length of square image patch
        self.batch_size = 15
        self.val_batch_size = 6
        self.plot_test_batch_size = 10

        self.showdata = False
        self.data_dir = "./datasets" # Directory of images

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.

        self.saveModel = True
        self.variationalTranslation = 0

        ### -------------------- ###
        ###  TRAINING PARAMETERS ###
        ### -------------------- ###

        # Starting epoch
        self.starting_epoch = 0
        # Number of epochs
        self.num_epochs  = 30
        # Number of iterations per epoch
        self.iter_per_epoch = 100
        # Number of training steps
        self.num_steps    = int(self.num_epochs*self.iter_per_epoch)

        # OPTIMIZER (AdamW)
        # Learning rate
        self.training_opt_lr = 0.001
        # Weight decay
        self.training_opt_weight_decay = 0.05
        # Optimizer Epsilon
        self.training_opt_eps = 1e-8
        # Optimizer Betas
        self.training_opt_betas = (0.9, 0.999)

        # SCHEDULER
        # Warm up epochs
        self.training_sch_warmup_epochs = 20
        # Decay epochs
        self.training_sch_decay_epochs  = 30
        # Scheduler decay rate
        self.training_sch_lr_decay_rate = 0.1
        # Number of warm-up steps
        self.training_sch_warmup_steps = int(self.training_sch_warmup_epochs*self.iter_per_epoch)
        # Number of decay steps
        self.training_sch_decay_steps  = int(self.training_sch_decay_epochs*self.iter_per_epoch)
        # Minimum learning rate
        self.training_sch_minimum_lr   = 5e-6
        # Warm-up learning rate
        self.training_sch_warmup_lr    = 5e-7
