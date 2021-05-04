import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func

IMAGE = 784
HIDDEN = 256
LATENT = 256
BATCH_SIZE = 100

class Generator (nn.Module):
    def __init__ (self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(LATENT, HIDDEN)
        self.bn1 = nn.BatchNorm1d(HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, HIDDEN)
        self.bn2 = nn.BatchNorm1d(HIDDEN)
        self.linear3 = nn.Linear(HIDDEN, HIDDEN)
        self.bn3 = nn.BatchNorm1d(HIDDEN)
        self.linear4 = nn.Linear(HIDDEN, IMAGE)

    def forward (self, Z):
        layer1 = Func.relu(self.bn1(self.linear1(Z)))
        layer2 = Func.relu(self.bn2(self.linear2(layer1)))
        layer3 = Func.relu(self.bn3(self.linear3(layer2)))
        layer4 = torch.sigmoid(self.linear4(layer3))
        return layer4

class Discriminator (nn.Module):
    def __init__ (self):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(IMAGE, HIDDEN)
        self.bn1 = nn.BatchNorm1d(HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, HIDDEN)
        self.bn2 = nn.BatchNorm1d(HIDDEN)
        self.linear3 = nn.Linear(HIDDEN, 1)

    def forward (self, X):
        layer1 = Func.relu(self.bn1(self.linear1(self.dropout(X))))
        layer2 = Func.relu(self.bn2(self.linear2(layer1)))
        layer3 = torch.clamp(torch.sigmoid(self.linear3(layer2)), 1e-10, 1-1e-10)
        return layer3

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
    #optimizerG = optim.SGD(gan.generator.parameters(), lr=5e-2, momentum=0.5)
    #optimizerD = optim.SGD(gan.discriminator.parameters(), lr=1e-2, momentum=0.5)
    optimizerG = optim.Adam(gan.generator.parameters(), lr=5e-4)
    optimizerD = optim.Adam(gan.discriminator.parameters(), lr=1e-4)
    idxs = torch.from_numpy(np.arange(X.shape[0])).to(device)

    criterion = nn.BCELoss()
    for it in range(1, 50001):  # epochs
        # Train discriminator
        for k in range(1):
            z = torch.randn(BATCH_SIZE, LATENT).to(device)
            fake = gan.generator(z)
            batchIdxs = idxs[torch.randint(len(idxs), (BATCH_SIZE,)).to(device)]
            real = X[batchIdxs]
            optimizerD.zero_grad()
            y = torch.cat((torch.zeros(BATCH_SIZE), torch.ones(BATCH_SIZE))).to(device)
            yhat = torch.cat((gan.discriminator(fake), gan.discriminator(real))).flatten()
            loss = criterion(yhat, y)
            loss.backward()
            optimizerD.step()
            dloss = loss.detach()

        # Train generator
        for k in range(1):
            z = torch.randn(BATCH_SIZE, LATENT).to(device)
            fake = gan.generator(z)
            optimizerG.zero_grad()
            yhat = gan.discriminator(fake)
            loss = torch.log(1 - yhat).mean()
            loss.backward()
            optimizerG.step()
            gloss = loss.detach()

        if it % 100 == 0:
            print("Iteration {}:  Dloss={}   Gloss={}".format(it, dloss, gloss))
            images = gan.generate(torch.randn(16, LATENT).to(device)).cpu().detach().numpy()
            volume = np.reshape(images, (images.shape[0], int(IMAGE**.5), int(IMAGE**.5)))
            sheet = np.hstack([ volume[i,:,:] for i in range(images.shape[0]) ])
            np.save("sheet.npy", sheet)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = GAN().to(device)
    
    X = torch.from_numpy(np.load("mnist.npy")).float().to(device)
    train(device, gan, X)
