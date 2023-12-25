class hyperParams:
    def __init__(self):
        # Constance hyperparameters. They have been tested and don't need to be tuned.
        self.NUM_TOKENS = 1003
        self.NUM_HEADS = 32
        self.EMBEDDING_DIM_ENCODE = self.NUM_HEADS
        self.EMBEDDING_DIM_DECODE = self.EMBEDDING_DIM_ENCODE * 3
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3
        self.DROPOUT_P = 1e-5
        self.LEARNING_RATE = 0.0001
        self.WEIGHT_DECAY = 1e-8
        self.NORM_FIRST = True
        self.ANNEALING_STEP_SIZE = 100
        self.ANNEALING_DECAY = 0.95