import os
import torch

class BaseOptions:
    """Configuration class for Pix2PixHD. Use directly in notebooks without argparse."""
    
    def __init__(self):
        # Experiment specifics
        self.name = 'pix2pixHD_experiment'
        self.gpu_ids = []
        self.checkpoints_dir = './checkpoints'
        self.model = 'pix2pixHD'
        self.norm = 'instance'
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False
        self.fp16 = False
        self.local_rank = 0
        self.isTrain = True
        
        # Input/output sizes
        self.batchSize = 1
        self.loadSize = 1024
        self.fineSize = 512
        self.label_nc = 35
        self.input_nc = 3
        self.output_nc = 3
        
        # For setting inputs
        self.dataroot = './datasets/cityscapes/'
        self.resize_or_crop = 'scale_width'
        self.serial_batches = False
        self.no_flip = False
        self.nThreads = 2
        self.max_dataset_size = float("inf")
        
        # For displays
        self.display_winsize = 512
        self.tf_log = False
        
        # For generator
        self.netG = 'global'
        self.ngf = 64
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0
        
        # For instance-wise features
        self.no_instance = False
        self.instance_feat = False
        self.label_feat = False
        self.feat_num = 3
        self.load_features = False
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 10
    
    def parse(self, save=True):
        """For compatibility: just returns self and optionally saves config."""
        str_ids = str(self.gpu_ids).split(',') if isinstance(self.gpu_ids, str) else [str(id) for id in self.gpu_ids]
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        
        # Set GPU IDs
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
        
        # Print options
        print('------------ Options -------------')
        for k, v in sorted(vars(self).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        # Save to disk
        if save and not getattr(self, 'continue_train', False):
            expr_dir = os.path.join(self.checkpoints_dir, self.name)
            os.makedirs(expr_dir, exist_ok=True)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(vars(self).items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        
        return self
