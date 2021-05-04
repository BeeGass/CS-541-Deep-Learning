import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
import matplotlib.pyplot as plt

INPUT = 784
HIDDEN = 512
LATENT = 64
BATCH_SIZE = 100

class Encoder (nn.Module):
    def __init__ (self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(INPUT, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, LATENT*2)
        self.dropout = nn.Dropout(0.05)

    def forward (self, X):
        layer1 = torch.tanh(self.linear1(self.dropout(X)))
        layer2 = self.linear2(layer1)
        return layer2

class Decoder (nn.Module):
    def __init__ (self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(LATENT, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, INPUT)

    def forward (self, Z):
        layer1 = torch.tanh(self.linear1(Z))
        layer2 = torch.sigmoid(self.linear2(layer1))
        return layer2

class VAE (nn.Module):
    def __init__ (self):
        super(VAE, self).__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    def z (self, x, eps):
        code = self.encoder(x)
        mu = code[:, 0:LATENT]  # First half of encoder's output
        sigma2 = torch.exp(code[:, LATENT:])  # Second half of encoder's output
        sigma = sigma2 ** 0.5
        z = mu + sigma * eps
        return z, mu, sigma2

    def x (self, z):
        x = self.decoder(z)
        return x

def showImages (images, num):
    images = images.cpu().detach().numpy()
    volume = np.reshape(images, (images.shape[0], int(INPUT**.5), int(INPUT**.5)))
    sheet = np.hstack([ volume[i,:,:] for i in range(images.shape[0]) ])
    np.save("sheet{}.npy".format(num), sheet)
    plt.imshow(sheet, cmap='gray')
    plt.show()

def showReconstructions (vae, z, num):
    xhat = vae.x(z)
    showImages(xhat, num)
    
def train (device, vae, X):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    idxs = np.arange(X.shape[0])

    def computeLoss (x):
        optimizer.zero_grad()
        eps = torch.randn(x.shape[0], LATENT).to(device)
        criterion = nn.BCELoss()
        z, mu, sigma2 = vae.z(x, eps)
        xhat = vae.x(z)
        KLDivergence = torch.sum(1 + torch.log(sigma2) - mu**2 - sigma2)
        loss = -1./2 * KLDivergence + 75*(X.shape[0] / BATCH_SIZE) * criterion(xhat, x)
        return loss, z

    for e in range(150):  # epochs
        print("Epoch {}".format(e))
        for i in np.arange(0, len(idxs), BATCH_SIZE):
            batchIdxs = torch.from_numpy(idxs[i:i+BATCH_SIZE])
            loss, z = computeLoss(X[batchIdxs])
            loss.backward()
            optimizer.step()
        if e % 5 == 0:
            print("loss={}".format(loss))
            showImages(X[batchIdxs[0:16]], 0)
            showReconstructions(vae, z[0:16], 1)
            showReconstructions(vae, torch.randn(16, LATENT).to(device), 2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    
    X = torch.from_numpy(np.load("mnist.npy")).float().to(device)
    train(device, vae, X)

