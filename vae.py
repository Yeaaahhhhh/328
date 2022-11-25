"""Assignment 9
Part 1: Variational Autoencoder + Conditional Variational Autoencoder

NOTE: Feel free to check: https://arxiv.org/pdf/1512.09300.pdf

NOTE: Write Down Your Info below:

    Name:

    CCID:

    Average Reconstruction Loss per Sample over Cifar10 Test Set: 0.6825 (vae) & 0.6837 (cvae)


"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ssim import SSIM



def compute_score(loss, min_thres, max_thres):
    if loss <= min_thres:
        base_score = 100.0
    elif loss >= max_thres:
        base_score = 0.0
    else:
        base_score = (1 - float(loss - min_thres) / (max_thres - min_thres)) \
                     * 100
    return base_score

# -----
# VAE Build Blocks

# #####
# TODO: Complete the encoder architecture
# #####


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.feat_dim = 128 * 8 * 8
        # #####
        # TODO: Complete the encoder architecture to calculate mu and log_var
        # mu and log_var will be used as inputs for the Reparameterization Trick,
        # generating latent vector z we need
        # #####

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # [, 16, 96, 96]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # [, 32, 96, 96]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [, 16, 48, 48]
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # [, 32, 48, 48]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # [, 64, 48, 48]
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # [, 64, 24, 24]
        )

        self.mu = nn.Linear(in_features=self.feat_dim, out_features=self.latent_dim)
        self.logvar = nn.Linear(in_features=self.feat_dim, out_features=self.latent_dim)


    
    def forward(self, x):
        # #####
        # TODO: Complete the encoder architecture to calculate mu and log_var
        # #####
        x = self.layer(x)
        x = x.view(-1, self.feat_dim)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


# #####
# TODO: Complete the decoder architecture
# #####

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.feat_dim = 128 * 8 * 8
        # #####
        # TODO: Complete the decoder architecture to reconstruct image from latent vector z
        # #####
        self.fc1 = nn.Linear(self.latent_dim, self.feat_dim)

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), # [, 32, 24, 24]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # [, 32, 48, 48]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [, 16, 96, 96]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), # [, 3, 96, 96]
            nn.Sigmoid(),
        )
        
    def forward(self, z):
        # #####
        # TODO: Complete the decoder architecture to reconstruct image xg from latent vector z
        # #####
        z = F.relu(self.fc1(z))
        z = z.view(-1, 128, 8, 8)
        z = self.layer(z)

        return z


# #####
# Wrapper for Variational Autoencoder
# #####

class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        # #####
        # TODO: Complete forward for VAE
        # #####
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xg = self.decode(z)

        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """


        return xg, mu, logvar, z

    def generate(
        self,
        n_samples: int,
        y=None
        ):
        # #####
        # TODO: Complete generate method for VAE
        # #####
        """Randomly sample from the latent space and return
        the reconstructed samples.
        Returns:
            xg: reconstructed image
            None: a placeholder simply.
        """


        mu = torch.randn((n_samples, self.latent_dim)).cuda()
        var_log = torch.randn((n_samples, self.latent_dim)).cuda()


        z = self.reparameterize(mu, var_log)
        xg = self.decode(z)
        return xg, None


# #####
# Wrapper for Conditional Variational Autoencoder
# #####

class CVAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        num_classes: int = 10,
        img_size: int = 32,
        ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        # #####
        # TODO: Insert additional layers here to encode class information
        # Feel free to change parameters for encoder and decoder to suit your strategy
        # #####
        self.encode = Encoder(latent_dim=latent_dim, in_channels=3)
        self.decode = Decoder(latent_dim=latent_dim+len(classes))



    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        # #####
        # TODO: Complete forward for CVAE
        # Note that you need to process label information HERE.
        # #####
        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        y_onehot = F.one_hot(y.view(-1), len(classes)).to(torch.float).cuda()
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat((z, y_onehot), dim=1)
        xg = self.decode(z)
        return xg, mu, log_var, z


    def generate(
        self,
        n_samples: int,
        y: torch.Tensor = None,
        ):
        # #####
        # TODO: Complete generate for CVAE
        # #####
        """Randomly sample from the latent space and return
        the reconstructed samples.
        NOTE: Randomly generate some classes here, if not y is provided.
        Returns:
            xg: reconstructed image
            y: classes for xg. 
        """

        z = torch.randn((n_samples, self.latent_dim)).cuda()
        y_onehot = F.one_hot(y.view(-1), len(classes)).to(torch.float).cuda()
        z = torch.cat((z, y_onehot), dim=1)
        xg = self.decode(z)
        y = None
        return xg, y


# #####
# Wrapper for KL Divergence
# #####

class KLDivLoss(nn.Module):
    def __init__(
        self,
        lambd: float = 1.0,
        ):
        super(KLDivLoss, self).__init__()
        self.lambd = lambd

    def forward(
        self, 
        mu, 
        log_var,
        ):
        loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
        self.lambd = min(0.001, self.lambd)
        return self.lambd * torch.mean(loss)


# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
# NOTE: DO NOT TRAIN IT LONGER THAN 100 EPOCHS.
batch_size = 256
workers = 2
latent_dim = 128
lr = 0.0005
num_epochs = 60
validate_every = 20
print_every = 1

conditional = True   # Flag to use VAE or CVAE

if conditional:
    name = "cvae"
else:
    name = "vae"

# Set up save paths
if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
    os.makedirs(os.path.join(os.path.curdir, "visualize", name))
save_path = os.path.join(os.path.curdir, "visualize", name)
ckpt_path = name + '.pt'


# TODO: Set up KL Annealing
kl_annealing = [i*0.001/(num_epochs-1) for i in range(num_epochs)]      # KL Annealing


# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
    transforms.ToTensor(),
    ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=tfms)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms,
    )

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)

subset = torch.utils.data.Subset(
    test_dataset, 
    [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

loader = torch.utils.data.DataLoader(
    subset, 
    batch_size=10)

# -----
# Model
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)

# -----
# Losses
# #####
# TODO: Initialize your loss criterions HERE.
# #####

KLDiv_criterion = KLDivLoss()
SSIM_criterion = SSIM()




best_total_loss = float("inf")

# Send to GPU
if torch.cuda.is_available():
    model = model.cuda()



optimizer = optim.Adam(model.parameters(), lr=lr)

# To further help with training
# NOTE: You can remove this if you find this unhelpful
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, [40, 50], gamma=0.1, verbose=False)


# -----
# Train loop

# #####
# TODO: Complete train_step for VAE/CVAE
# #####

def train_step(x, y):
    """One train step for VAE/CVAE.
    You should return average total train loss(sum of reconstruction losses, kl divergence loss)
    and all individual average reconstruction loss (l2, bce, ssim) per sample.
    Args:
        x, y: one batch (images, labels) from Cifar10 train set.
    Returns:
        loss: total loss per batch.
        l2_loss: MSE loss for reconstruction.
        bce_loss: binary cross-entropy loss for reconstruction.
        ssim_loss: ssim loss for reconstruction.
        kldiv_loss: kl divergence loss.
    """



    if conditional:
        optimizer.zero_grad()
        x_, mu, logvar, z = model(x, y)
        mse_loss = F.mse_loss(x_, x)
        bce_loss = F.binary_cross_entropy(x_, x, size_average=True)
        ssim_loss = 1-SSIM_criterion(x_, x)
        kldiv_loss = KLDiv_criterion(x_, x)

        loss_recon =  mse_loss + ssim_loss + bce_loss
        loss_vae = loss_recon + kldiv_loss
        loss_vae.backward()
        optimizer.step()
        return loss_vae, mse_loss, bce_loss, ssim_loss, kldiv_loss
    else:
        optimizer.zero_grad()
        x_, mu, logvar, z = model(x, y)
        mse_loss = F.mse_loss(x_, x)
        bce_loss = F.binary_cross_entropy(x_, x, size_average=True)
        ssim_loss = 1-SSIM_criterion(x_, x)
        kldiv_loss = KLDiv_criterion(x_, x)

        loss_recon =  mse_loss + ssim_loss + bce_loss
        loss_vae = loss_recon + kldiv_loss
        loss_vae.backward()
        optimizer.step()
        return loss_vae, mse_loss, bce_loss, ssim_loss, kldiv_loss


def denormalize(x):
    """Denomalize a normalized image back to uint8.
    Args:
        x: torch.Tensor, in [0, 1].
    Return:
        x_denormalized: denormalized image as numpy.uint8, in [0, 255].
    """
    # #####
    # TODO: Complete denormalization.
    # #####
    x = x.swapaxes(1,3)
    x = 255 * x.detach().cpu().numpy()
    x = x.astype(np.uint8)
    return x

# Loop HERE
l2_losses = []
bce_losses = []
ssim_losses = []
kld_losses = []
total_losses_test = []

total_losses_train = []

for epoch in range(1, num_epochs + 1):
    total_loss_train = 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # Train step
        model.train()
        loss, recon_loss, bce_loss, ssim_loss, kldiv_loss = train_step(x, y)

        total_loss_train += loss.cpu() * x.shape[0]
        
        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(epoch, i, loss, recon_loss, ssim_loss, bce_loss, kldiv_loss))

    total_losses_train.append(total_loss_train / len(train_dataset))

    # Test loop
    if epoch % validate_every == 0:
        # Loop through test set
        model.eval()

        # TODO: Accumulate average reconstruction losses per sample individually for plotting
        # Feel free to add code wherever you want to accumulate the loss
        total_loss_test = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                xg, mu, log_var, _ = model(x, y)
                # TODO: Accumulate average reconstruction losses per batch individually for plotting
                mse_loss = F.mse_loss(xg, x)
                bce_loss = F.binary_cross_entropy(xg, x, size_average=True)
                ssim_loss = 1 - SSIM_criterion(xg, x)
                kldiv_loss = KLDiv_criterion(xg, x)
                loss_recon = mse_loss + ssim_loss + bce_loss
                loss_vae = loss_recon + kldiv_loss

                total_loss_test += loss_vae.cpu() * x.shape[0]


                avg_total_recon_loss_test = loss_recon

                l2_losses.append(mse_loss.cpu())
                bce_losses.append(bce_loss.cpu())
                ssim_losses.append(ssim_loss.cpu())
                kld_losses.append(kldiv_loss.cpu())

            total_losses_test.append(total_loss_test / len(test_dataset))

            # Plot losses
            if epoch > 1:
                plt.plot(l2_losses, label="L2 Reconstruction")
                plt.plot(bce_losses, label="BCE")
                plt.plot(ssim_losses, label="SSIM")
                plt.plot(kld_losses, label="KL Divergence")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim([1, epoch])
                plt.legend()
                plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
                plt.clf()
                plt.close('all')

                plt.plot(total_losses_test, label="Total Loss Test")
                plt.plot(total_losses_train, label="Total Loss Train")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim([1, epoch])
                plt.legend()
                plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
                plt.clf()
                plt.close('all')

            # Save best model
            if avg_total_recon_loss_test < best_total_loss and epoch > 50:
                torch.save(model.state_dict(), ckpt_path)
                best_total_loss = avg_total_recon_loss_test
                print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(best_total_loss))

        # Do some reconstruction
        model.eval()
        with torch.no_grad():
            x, y = next(iter(loader))
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            #y_onehot = F.one_hot(y, 10).float()
            xg, _, _, _ = model(x, y)

            # Visualize
            xg = denormalize(xg)
            x = denormalize(x)

            y = y.cpu().numpy()

            plt.figure(figsize=(10, 5))
            for p in range(10):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.text(0, 0, "fake", color='black',
                            backgroundcolor='white', fontsize=8)
                plt.subplot(4, 5, p + 1 + 10)
                plt.imshow(x[p])
                plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')
            print("Figure saved at epoch {}.".format(epoch))

    # #####
    # TODO: Complete KL-Annealing.
    # #####
    # KL Annealing
    # Adjust scalar for KL Divergence loss
    KLDiv_criterion.lambd = kl_annealing[epoch-1]

    print("Lambda:", KLDiv_criterion.lambd)
    
    # LR decay
    scheduler.step()
    
    # print()

# Generate some random samples
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)
if torch.cuda.is_available():
    model = model.cuda()
ckpt = torch.load(name+'.pt')
model.load_state_dict(ckpt)

# Generate 20 random images
y_random = torch.randint(high=len(classes), size=(40 ,1)).squeeze()
print(y_random.shape)
xg, y = model.generate(40, y_random)
xg = denormalize(xg)
if y is not None:
    y = y.cpu().numpy()

plt.figure(figsize=(20, 5))
for p in range(40):
    plt.subplot(8, 5, p+1)
    if y is not None:
        plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                 backgroundcolor='white', fontsize=8)
    plt.imshow(xg[p])
    plt.axis('off')

plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
plt.clf()
plt.close('all')

if conditional:
    min_val, max_val = 0.92, 1.0
else:
    min_val, max_val = 0.92, 1.0

print("Total reconstruction loss:", best_total_loss)
score = compute_score(best_total_loss, min_val, max_val)
print("Your Assignment Score:", score)