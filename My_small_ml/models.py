import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 3 - RGB 128 - Кол-во выходных пар, 5 размер ядра
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.conv2 = nn.Conv2d(128, 3, 3)
        self.padding1 = nn.ReflectionPad2d(2)
        self.padding2 = nn.ReflectionPad2d(1)


    def forward(self, x):
        # -> n, 3, 32, 32

        x = self.padding1(x)
        x = F.relu(self.conv1(x)) # -> n, 6, 14, 14
        x = self.padding2(x)
        x = self.conv2(x) # -> n, 16, 5, 5 Не делаем активацию чтобы корректировать ошибки при отрицательных чисел через градиет

        return x


class GridConvNet(nn.Module):
    def __init__(self, num_layers, Layers_sizes, kernal_size):

        super(GridConvNet, self).__init__()

        self.num_layers = num_layers
        self.Layers_sizes = Layers_sizes
        self.kernal_size = kernal_size

        # All Convolutions functions

        self.Conv = nn.ParameterList()

        if self.num_layers == 1:
            self.Conv.append(nn.Conv2d(3, 3, self.kernal_size))

        elif self.num_layers == 2:
            self.Conv.append(nn.Conv2d(3, self.Layers_sizes[0], self.kernal_size))
            self.Conv.append(nn.Conv2d(self.Layers_sizes[0], 3, self.kernal_size))

        else:
            self.Conv.append(nn.Conv2d(3, self.Layers_sizes[0], self.kernal_size))

            for i in range(self.num_layers - 2):
                self.Conv.append(nn.Conv2d(self.Layers_sizes[i], self.Layers_sizes[i+1], self.kernal_size))

            self.Conv.append(nn.Conv2d(self.Layers_sizes[self.num_layers - 3], 3, self.kernal_size))

        # All Pading functions

        self.Pad = nn.ParameterList()

        for i in range(self.num_layers):
            if self.kernal_size == 3:
                self.Pad.append(nn.ReflectionPad2d(1))
            elif self.kernal_size == 5:
                self.Pad.append(nn.ReflectionPad2d(2))
            elif self.kernal_size == 9:
                self.Pad.append(nn.ReflectionPad2d(4))


    def forward(self, x):
        # -> n, 3, 32, 32

        for i in range(self.num_layers - 1):
            x = self.Pad[i](x)
            x = self.Conv[i](x)
            x = F.relu(x)
        else:
            i = self.num_layers - 1
            x = self.Pad[i](x)
            x = self.Conv[i](x)

        return x