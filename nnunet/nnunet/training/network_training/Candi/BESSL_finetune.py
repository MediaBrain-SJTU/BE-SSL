from collections import OrderedDict
from nnunet.training.network_training.Candi.CandiBaseTrainer import CandiBaseTrainer
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from nnunet.network_architecture.basic_unet import unet

class preTrain_Candi_vox(CandiBaseTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False,ratio=1.0):
        super().__init__(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage, unpack_data=unpack_data, deterministic=deterministic, fp16=fp16)
        
        self.ratio = ratio

        self.output_folder = self.output_folder.replace('/fold', '/ratio_{}/fold'.format(str(self.ratio)))
        self.output_folder_base = self.output_folder

        self.pretrain_path = './nnunet/data/nnUNet_trained_models/nnUNet/'

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """

        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 32, 32, 16]
        self.network = unet(nf_enc, nf_dec,dim=3, num_classes = self.num_classes)

        checkpoint = torch.load(self.pretrain_path, map_location=torch.device('cpu'))['state_dict']

        new_state_dict = OrderedDict()
        for k, value in checkpoint.items():
            key = k
            if 'enc' in key:
                new_state_dict[key] = value
        
        self.network.load_state_dict(new_state_dict,strict=False)

        if torch.cuda.is_available():
            self.network.cuda()
