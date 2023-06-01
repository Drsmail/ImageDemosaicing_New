import datetime
import itertools
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from C_data_loader import MozaikDataset
import tqdm

from torchmetrics import MeanMetric
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
from models import GridConvNet

from torch.utils.tensorboard import SummaryWriter

import cProfile
import pstats



if __name__ == "__main__":

    with cProfile.Profile() as profile:

        logdir = "runs/tf_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = SummaryWriter(logdir)

        # GPU if possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device}")

        # Hyper-parameters
        num_epochs = 10
        batch_size = 16

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = MozaikDataset('F:/Unpaked_dataset/images/test', transform, split=1, trim=True)
        test_dataset = MozaikDataset('F:/Unpaked_dataset/images/val', transform, split=1)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2, persistent_workers=True)

        params = {

            'num_layers': [2, 3, 4, 5], #0
            'layer_1_out': [16],#1
            'layer_2_out': [16],#2
            'layer_3_out': [16],#3
            'layer_4_out': [16],#4
            'criterion': [torch.nn.L1Loss, torch.nn.MSELoss],#5
            'optimizer': [torch.optim.Adam, torch.optim.SGD],#6
            'learning_rate': [0.01, 0.001, 0.0001, 0.00001],#7
            'kenal_size': [3, 5, 9]#8

        }

        param_combinations = list(itertools.product(*params.values()))

        for run, Params in enumerate(param_combinations):

            if run == 2:
                break

            writer.add_text(f"Model par_m{run}",
                            f" num_layers: {Params[0]}, layer_1_out: {Params[1]} layer_2_out: {Params[2]} layer_3_out: {Params[3]} layer_4_out: {Params[4]} criterion: {Params[5]} optimizer: {Params[6]} learning_rate: {Params[7]} kenal_size: {Params[8]}")

            print(f" num_layers: {Params[0]}, layer_1_out: {Params[1]} layer_2_out: {Params[2]} layer_3_out: {Params[3]} layer_4_out: {Params[4]} criterion: {Params[5]} optimizer: {Params[6]} learning_rate: {Params[7]} kenal_size: {Params[8]}")

            learning_rate = Params[7]
            Layers_sizes = [Params[1], Params[2], Params[3], Params[4]]
            #  num_layers, Layers_sizes, kernal_size
            model = GridConvNet(Params[0], Layers_sizes, Params[8]).to(device)
            criterion = Params[5]()
            optimizer = Params[6](model.parameters(), lr=learning_rate)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # alt torch.optim.Adam()

            loss_train = MeanMetric()
            loss_valid = MeanMetric()
            PSNR_torch = PeakSignalNoiseRatio(reduction='elementwise_mean', dim=[0, 2, 3], data_range=1.0).to(device)
            SSIM_torch = StructuralSimilarityIndexMeasure(reduction='elementwise_mean', dim=[0, 2, 3], data_range=1.0).to(
                device)

            for epoch in range(num_epochs):

                psnr_train = 0
                ssim_train = 0

                for i, (images, target) in tqdm.tqdm(enumerate(train_loader)):
                    # input_layer:
                    images = images.to(device)
                    target = target.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, target)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        loss_train.update(loss.item())
                        psnr_train += PSNR_torch(outputs.detach(), target.detach())
                        ssim_train += SSIM_torch(outputs.detach(), target.detach())

                psnr_train = psnr_train / len(train_loader)
                ssim_train = ssim_train / len(train_loader)

                psnr_val = 0
                ssim_val = 0

                with torch.no_grad():
                    for i, (images, target) in tqdm.tqdm(enumerate(test_loader)):
                        images = images.to(device)
                        target = target.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, target)

                        loss_valid.update(loss.item())

                        psnr_val += PSNR_torch(outputs.detach(), target.detach())
                        ssim_val += SSIM_torch(outputs.detach(), target.detach())

                    else:

                        psnr_val = psnr_val / len(test_loader)
                        ssim_val = ssim_val / len(test_loader)

                        writer.add_scalars('PSNR', {f"train_m{run}": psnr_train,
                                                    f"valid_m{run}": psnr_val}, epoch + 1)

                        writer.add_scalars('SSIM', {f"train_m{run}": ssim_train,
                                                    f"valid_m{run}": ssim_val}, epoch + 1)

                        writer.add_scalars('Loss', {f"train_m{run}": loss_train.compute(),
                                                    f"valid_m{run}": loss_valid.compute()}, epoch + 1)

                        # loss_valid.reset()
                        # loss_train.reset()

                        writer.add_images(f"Train_m{run}/desired", target, epoch + 1)
                        writer.add_images(f"Train_m{run}/restored", outputs, epoch + 1)
                        writer.add_images(f"Valid_m{run}/desired", target, epoch + 1)
                        writer.add_images(f"Valid_m{run}/restored", outputs, epoch + 1)
                        torch.save(model.state_dict(), logdir + f"cnn_m{run}_e{epoch+1}.pth")

            # for start in terminal type
            # tensorboard --logdir=runs/tf_logs

        writer.flush()
        writer.close()

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()




