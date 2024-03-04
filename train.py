import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms

from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:

    def __init__(self, data_dict):

        self.results_dir = data_dict['results_dir']

        self.train_results_dir = os.path.join(self.results_dir, 'train')
        os.makedirs(self.train_results_dir, exist_ok=True)

        self.checkpoints_dir = data_dict['checkpoints_dir']

        self.data_dir = data_dict['data_dir']

        self.num_epoch = data_dict['num_epoch']
        self.batch_size = data_dict['batch_size']
        self.lr = data_dict['lr']

        self.num_freq_disp = data_dict['num_freq_disp']
        self.num_freq_save = data_dict['num_freq_save']

        self.train_continue = data_dict['train_continue']
        self.load_epoch = data_dict['load_epoch']

        # check if we have a gpu
        if torch.cuda.is_available():
            print("GPU is available")
            self.device = torch.device("cuda:0")
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")


    def save(self, dir_chck, net, optimG, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': net.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))
        

    def load(self, dir_chck, net, epoch, optimG=[]):

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        net.load_state_dict(dict_net['netG'])
        optimG.load_state_dict(dict_net['optimG'])

        return net, optimG, epoch
    

    def train(self):

        ### transforms ###

        print(self.data_dir)
        start_time = time.time()
        min, max = compute_global_min_max_and_save(self.data_dir)
        mean, std = compute_global_mean_and_std(self.data_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        transform_train = transforms.Compose([
            MinMaxNormalize(min, max),
            RandomCrop(output_size=(128,128)),
            RandomHorizontalFlip(),
            PPS(),
            ToTensorPPS()
        ])

        transform_inv_train = transforms.Compose([
            BackTo01Range(),
            ToNumpy()
        ])


        ### make dataset and loader ###

        dataset_train = PPSDataset(root_folder_path=self.data_dir, 
                                    transform=transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0)

        num_train = len(dataset_train)
        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))


        ### initialize network ###

        model = NewUNet().to(self.device)

        MSE_loss = nn.MSELoss().to(self.device)

        params = model.parameters()

        optimizer = torch.optim.Adam(params, self.lr, betas=(0.5, 0.999))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


        st_epoch = 0
        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            net, optimG, st_epoch = self.load(self.checkpoints_dir, model, self.load_epoch, optimizer)


        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            model.train()  # Set the network to training mode

            for batch, data in enumerate(loader_train, 1):
                
                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                
                input_stack_2x2, target_stack_2x2, input_stack_4x4, target_stack_4x4 = data

                output_stack_2x2 = model(input_stack_2x2)
                output_stack_4x4 = model(input_stack_4x4)

                optimizer.zero_grad()

                loss_2x2 = MSE_loss(output_stack_2x2, target_stack_2x2)
                loss_4x4 = MSE_loss(output_stack_4x4, target_stack_4x4)

                loss = 1 * loss_2x2 + 10 * loss_4x4

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()
                # scheduler.step()



                logging.info('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                             % (epoch, batch, num_batch_train, loss))

                
                if should(self.num_freq_disp):

                    input_stack_2x2 = transform_inv_train(input_stack_2x2)[..., 0]
                    target_stack_2x2 = transform_inv_train(target_stack_2x2)[..., 0]
                    output_stack_2x2 = transform_inv_train(output_stack_2x2)[..., 0]
                    input_stack_4x4 = transform_inv_train(input_stack_4x4)[..., 0]
                    target_stack_4x4 = transform_inv_train(target_stack_4x4)[..., 0]
                    output_stack_4x4 = transform_inv_train(output_stack_4x4)[..., 0]

                    # plot_intensity_distribution(output_stack_2x2)

                    for j in range(input_stack_2x2.shape[0]):
                        
                        plt.imsave(os.path.join(self.train_results_dir, f"{j}_i_2x2.png"), input_stack_2x2[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, f"{j}_t_2x2.png"), target_stack_2x2[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, f"{j}_o_2x2.png"), output_stack_2x2[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, f"{j}_i_4x4.png"), input_stack_4x4[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, f"{j}_t_4x4.png"), target_stack_4x4[j, :, :], cmap='gray')
                        plt.imsave(os.path.join(self.train_results_dir, f"{j}_o_4x4.png"), output_stack_4x4[j, :, :], cmap='gray')

                # Saving model checkpoint does not depend on saving images, so it remains the same.
                # if (epoch % self.num_freq_save) == 0:
                    self.save(self.checkpoints_dir, model, optimizer, epoch)
