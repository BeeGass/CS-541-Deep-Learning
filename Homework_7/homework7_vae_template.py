import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func

INPUT = 784
HIDDEN = 512
BATCH_SIZE = 100

class Encoder (nn.Module):
    def __init__ (self):
        super(Encoder, self).__init__()
        # TODO initialize layers
        pass

    def forward (self, X):
        # TODO execute layers and return result
        pass

class Decoder (nn.Module):
    def __init__ (self):
        super(Decoder, self).__init__()
        # TODO initialize layers
        pass

    def forward (self, Z):
        # TODO execute layers and return result
        pass

class VAE (nn.Module):
    def __init__ (self):
        super(VAE, self).__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    # Computes the hidden representation given the pre-sampled values eps
    def z (self, x, eps):
        code = self.encoder(x)
        # TODO extract mu and sigma2 from the code, and then use them and eps
        # to compute z.
        return z, mu, sigma2

    def x (self, z):
        x = self.decoder(z)
        return x

def train (device, vae, X):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)  # worked for me

    def computeLoss (x):
        # TODO implement your custom loss
        pass

    for e in range(150):  # epochs
        print("Epoch {}".format(e))
        for i in np.arange(0, len(X), BATCH_SIZE):
            # Call computeLoss on each minibatch
            loss, z = computeLoss(miniBatch)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    
    X = torch.from_numpy(np.load("mnist.npy")).float().to(device)
    train(device, vae, X)
