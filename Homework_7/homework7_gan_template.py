import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func

class Generator (nn.Module):
    def __init__ (self):
        super(Generator, self).__init__()
        # TODO initialize layers
        pass

    def forward (self, Z):
        # TODO execute layers and return result
        pass

class Discriminator (nn.Module):
    def __init__ (self):
        super(Discriminator, self).__init__()
        # TODO initialize layers
        pass

    def forward (self, X):
        # TODO execute layers and return result
        pass

class GAN (nn.Module):
    def __init__ (self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def generate (self, x):
        z = self.generator(x)
        return z

    def discriminate (self, z):
        return self.discriminator(z)

def train (device, gan, X):
    optimizerG = optim.Adam(gan.generator.parameters(), lr=5e-4)  # worked for me
    optimizerD = optim.Adam(gan.discriminator.parameters(), lr=1e-4)  # worked for me

    criterion = nn.BCELoss()
    for it in range(NUM_ITERATIONS):
        # TODO Train discriminator

        # TODO Train generator

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = GAN().to(device)
    
    X = torch.from_numpy(np.load("mnist.npy")).float().to(device)
    train(device, gan, X)
