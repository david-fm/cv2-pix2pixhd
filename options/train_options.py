from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """Training configuration for Pix2PixHD. Inherits from BaseOptions."""
    
    def __init__(self):
        super().__init__()
        self.isTrain = True
        
        # For displays
        self.display_freq = 100
        self.print_freq = 100
        self.save_latest_freq = 1000
        self.save_epoch_freq = 10
        self.no_html = False
        self.debug = False
        
        # For training
        self.continue_train = False
        self.load_pretrain = ''
        self.which_epoch = 'latest'
        self.phase = 'train'
        self.niter = 100
        self.niter_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        
        # For discriminators
        self.num_D = 2
        self.n_layers_D = 3
        self.ndf = 64
        self.lambda_feat = 10.0
        self.no_ganFeat_loss = False
        self.no_vgg_loss = False
        self.no_lsgan = False
        self.pool_size = 0
