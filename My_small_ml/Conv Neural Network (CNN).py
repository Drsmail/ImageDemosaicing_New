import datetime
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from C_data_loader import MozaikDataset
import tqdm

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import MeanMetric
from models import ConvNet

from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":

    logdir = "runs/tf_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(logdir)

    # GPU if can
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # Hyper-parameters
    num_epochs = 5
    batch_size = 16
    learning_rate = 0.001

    # dataset has PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = MozaikDataset('F:/Unpaked_dataset/images/train', transform, split=10, trim=False)

    test_dataset = MozaikDataset('F:/Unpaked_dataset/images/val', transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2, persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2, persistent_workers=True)


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        # torch utils save_image?


    # get some random training images
    dataiter = iter(train_loader)
    example_images, target = next(dataiter)

    # show images

    input_img_grid = torchvision.utils.make_grid(example_images)

    writer.add_image("input images", input_img_grid)

    # imshow(torchvision.utils.make_grid(images))

    model = ConvNet().to(device)
    criterion = torch.nn.MSELoss()  # alt nn.L1Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # alt torch.optim.Adam()

    writer.add_graph(model, example_images.to(device))

    n_total_steps = len(train_loader)

    print(f"Starting training: total iteration to do is {len(train_loader)}")

    torch.cuda.empty_cache()
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


                psnr_val += PSNR_torch(outputs.detach(), target.detach())
                ssim_val += SSIM_torch(outputs.detach(), target.detach())

            else:

                loss_valid.update(loss.item())
                psnr_val = psnr_val / len(test_loader)
                ssim_val = ssim_val / len(test_loader)

                writer.add_scalars('PSNR', {'train': psnr_train,
                                            'valid': psnr_val}, epoch + 1)

                writer.add_scalars('SSIM', {'train': ssim_train,
                                            'valid': ssim_val}, epoch + 1)

                writer.add_scalars('Loss', {'train': loss_train.compute(),
                                            'valid': loss_valid.compute()}, epoch + 1)

                # loss_valid.reset()
                # loss_train.reset()


                writer.add_images('Train/desired', target, epoch + 1)
                writer.add_images('Train/restored', outputs, epoch + 1)
                writer.add_images('Valid/desired', target, epoch + 1)
                writer.add_images('valid/restored', outputs, epoch + 1)

                torch.save(model.state_dict(), logdir + f"cnn{epoch+1}.pth")

    # for start in terminal type
    # tensorboard --logdir=runs/tf_logs
    # tensorboard --logdir=My_small_ml/runs/tf_logs

    writer.flush()
    writer.close()




