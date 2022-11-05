from collections import OrderedDict
from nnunet.training.network_training.Candi.CandiBaseTrainer import CandiBaseTrainer
from nnunet.training.loss_functions.reg_loss import NCC, MSE, Grad
import torch
import torch.backends.cudnn as cudnn
from batchgenerators.utilities.file_and_folder_operations import *
from time import time, sleep
import numpy as np
import torch.nn as nn
from nnunet.network_architecture.initialization import InitWeights_He
import os
from nnunet.network_architecture.SpatialTransformer import SpatialTransformer
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from nnunet.network_architecture.basic_unet_2dec import unet
from nnunet.training.dataloading.edt_dataset_loading_Candi import DataLoader3D_edt


class Candi_preTrain(CandiBaseTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage, unpack_data=unpack_data, deterministic=deterministic, fp16=fp16)
        self.sim_loss_fn = MSE().loss
        self.grad_loss_fn = Grad().loss
        self._lambda = 1e-2
        self.num_classes = 3
        self.initial_lr = 3e-4
        self.num_batches_per_epoch = 200
        self.max_num_epochs = 100
        self.ratio = 1.0

        self.atlas_path = './nnunet/data/nnUNet_preprocessed/Task160_CandiBrainSegmentation/nnUNetData_plans_v2.1_stage0/025.npy'

        self._beta = 1

    def run_training(self):
        self.save_debug_information()
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            l_grad_epoch = []
            l_sim_epoch = []
            l_edt_epoch= []

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l, l_grad, l_sim, l_edt = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
                        l_grad_epoch.append(l_grad)
                        l_sim_epoch.append(l_sim)
                        l_edt_epoch.append(l_edt)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l, l_grad, l_sim, l_edt  = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)
                    l_grad_epoch.append(l_grad)
                    l_sim_epoch.append(l_sim)
                    l_edt_epoch.append(l_edt)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : {} l_grad: {} l_sim loss: {} l_edt loss: {}".format(self.all_tr_losses[-1], np.mean(l_grad_epoch), np.mean(l_sim_epoch), np.mean(l_edt_epoch)))

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))


    def on_epoch_end(self):

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        return True


    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """

        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 32, 32, 16]
        vol_size = (128,160,160)
        self.network = unet(nf_enc, nf_dec,dim=3, num_classes = 3)
        self.trf = SpatialTransformer(vol_size)

        self.atlas = np.load(self.atlas_path)[0][np.newaxis, np.newaxis,:,:,:]
        self.atlas = np.repeat(self.atlas, self.batch_size, axis=0)
        if torch.cuda.is_available():
            self.network.cuda()
            self.trf.cuda()
            self.atlas = to_cuda(maybe_to_torch(self.atlas))
    

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()


        if self.fp16:
            with autocast():
                pred_edt,flow = self.network(data)
                warp = self.trf(data, flow)
                l_grad = self._lambda * self.grad_loss_fn(None, flow)
                l_sim = self.sim_loss_fn(warp, self.atlas) 

                l_edt = self._beta * self.sim_loss_fn(pred_edt, target)

                del data
                l = l_grad + l_sim + l_edt

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            flow = self.network(data)
            warp = self.trf(data, flow)
            l_grad = self._lambda * self.grad_loss_fn(None, flow)
            l_sim = self.sim_loss_fn(warp, self.atlas) 

            del data
            l = l_grad + l_sim

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        del target

        return l.detach().cpu().numpy(),l_grad.detach().cpu().numpy(), l_sim.detach().cpu().numpy(), l_edt.detach().cpu().numpy()


    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)
        self.plans['plans_per_stage'][0]['batch_size']=2
        self.plans['plans_per_stage'][0]['patch_size']=np.array([128,160,160])


    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D_edt(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D_edt(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val