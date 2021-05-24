class Config():
    def __init__(self):
        self.input_size = 224 # Side length of square image patch
        self.batch_size = 6
        self.val_batch_size = 4
        self.test_batch_size = 1

        self.num_epochs = 32 #250 for real
        self.data_dir = "./datasets" # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        
        self.saveModel = True
