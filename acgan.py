"""Assignment 9
Part 2: AC-GAN

NOTE: Feel free to check: https://arxiv.org/pdf/1610.09585.pdf

NOTE: Write Down Your Info below:

    Name: Leslie Qin

    CCID: xq4

    Auxiliary Test Accuracy on Cifar10 Test Set: 0.6752


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
from torch.autograd import Variable
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


# -----
# AC-GAN Build Blocks

# #####
# TODO: Complete the generator architecture
# #####

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        self.fc1 = nn.Linear(latent_dim + num_classes, 384)

        self.t1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=(4, 4), stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.t2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.t3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.t4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=3, kernel_size=(4, 4), stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z): # we concat noise 'z' with its fake label 'y' before feeding them to generator !!
        # #####
        # TODO: Complete the generator architecture
        # #####
        x = self.fc1(z)
        x = x.view(-1, 384, 1, 1)
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)

        return x



# #####
# TODO: Complete the Discriminator architecture
# #####

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc_source = nn.Linear(4 * 4 * 512, 1)
        self.fc_class = nn.Linear(4 * 4 * 512, 10)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax()

    def forward(self, x):
        # #####
        # TODO: Complete the discriminator architecture
        # #####
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = x.view(-1, 4 * 4 * 512)
        rf = self.sig(
            self.fc_source(x)).view(-1)  # checks source of the data---i.e.--data generated(fake) or from training set(real)
        c = self.soft(self.fc_class(
            x))  # checks class(label) of data--i.e. to which label the data belongs in the CIFAR10 dataset
        return rf, c

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


    # -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
batch_size = 256
workers = 6
latent_dim = 100
lr = 0.0002
num_epochs = 300
validate_every = 1
print_every = 100

save_path = os.path.join(os.path.curdir, "visualize", "gan")
if not os.path.exists(os.path.join(os.path.curdir, "visualize", "gan")):
    os.makedirs(os.path.join(os.path.curdir, "visualize", "gan"))
ckpt_path = 'acgan.pt'

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=tfms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)


# -----
# Model
# #####
# TODO: Initialize your models HERE.
# #####
gen = Generator(latent_dim=latent_dim, out_channels=3, num_classes=10).cuda()
disc = Discriminator(in_channels=3, num_classes=10).cuda()
gen.apply(weights_init)

# -----
# Losses

# #####
# TODO: Initialize your loss criterion.
# #####

adv_loss = torch.nn.BCELoss()
aux_loss = torch.nn.NLLLoss()

if torch.cuda.is_available():
    gen = gen.cuda()
    disc = disc.cuda()
    adv_loss = adv_loss.cuda()
    aux_loss = aux_loss.cuda()

# Optimizers for Discriminator and Generator, separate

# #####
# TODO: Initialize your optimizer(s).
# #####

optimD = torch.optim.Adam(disc.parameters(), lr=lr)
optimG = torch.optim.Adam(gen.parameters(), lr=lr)


# -----
# Train loop

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


# For visualization part
# Generate 20 random sample for visualization
# Keep this outside the loop so we will generate near identical images with the same latent featuresper train epoch

# #####
# TODO: Complete train_step for AC-GAN
# #####

real_label = torch.FloatTensor(batch_size).cuda()
real_label.fill_(1)
# real_label = real_label.unsqueeze(1)

fake_label = torch.FloatTensor(batch_size).cuda()
fake_label.fill_(0)
# fake_label = fake_label.unsqueeze(1)

eval_noise = torch.FloatTensor(batch_size, 110).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (batch_size, 110))
eval_label = np.random.randint(0, 10, batch_size)
eval_onehot = np.zeros((batch_size, 10))
eval_onehot[np.arange(batch_size), eval_label] = 1
eval_noise_[np.arange(batch_size), :10] = eval_onehot[np.arange(batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(batch_size, 110))
eval_noise = eval_noise.cuda()


random_z = eval_noise
random_y = eval_label
print(random_z.shape)
print(random_y.shape)


def compute_acc(preds, labels):
	correct = 0
	preds_ = preds.data.max(1)[1]
	correct = preds_.eq(labels.data).cpu().sum()
	acc = float(correct) / float(len(labels.data)) * 100.0
	return acc


def train_step(x, y):
    """One train step for AC-GAN.
    You should return loss_g, loss_d, acc_d, a.k.a:
        - average train loss over batch for generator
        - average train loss over batch for discriminator
        - auxiliary train accuracy over batch
    """
    optimD.zero_grad()
    image, label = x.cuda(), y.cuda()
    source_, class_ = disc(image)  # we feed the real images into the discriminator
    # print(source_.size())
    source_error = adv_loss(source_, real_label)  # label for real images--1; for fake images--0
    class_error = aux_loss(class_, label)
    error_real = source_error + class_error
    error_real.backward()
    optimD.step()

    accuracy = compute_acc(class_, label)  # getting the current classification accuracy

    # training with fake data now----

    noise = torch.FloatTensor(batch_size, 110).normal_(0, 1)
    noise_ = np.random.normal(0, 1, (batch_size, 110))
    label = np.random.randint(0, 10, batch_size)
    label_onehot = np.zeros((batch_size, 10))
    label_onehot[np.arange(batch_size), label] = 1
    noise_[np.arange(batch_size), :10] = label_onehot[np.arange(batch_size)]
    noise_ = (torch.from_numpy(noise_))
    noise.data.copy_(noise_.view(batch_size, 110))
    noise = noise.cuda()
    label = ((torch.from_numpy(label)).long())
    label = label.cuda()  # converting to tensors in order to work with pytorch
    noise_image = gen(noise)
    # print(noise_image.size())

    source_, class_ = disc(noise_image.detach())  # we will be using this tensor later on
    # print(source_.size())
    source_error = adv_loss(source_, fake_label)  # label for real images--1; for fake images--0
    class_error = aux_loss(class_, label)
    error_fake = source_error + class_error
    error_fake.backward()
    optimD.step()

    '''
    Now we train the generator as we have finished updating weights of the discriminator
    '''

    gen.zero_grad()
    source_, class_ = disc(noise_image)
    source_error = adv_loss(source_,
                              real_label)  # The generator tries to pass its images as real---so we pass the images as real to the cost function
    class_error = aux_loss(class_, label)
    error_gen = source_error + class_error
    error_gen.backward()
    optimG.step()

    return error_gen, (error_real+error_fake)/2, accuracy


def test(
    test_loader,
    ):
    """Calculate accuracy over Cifar10 test set.
    """
    size = len(test_loader.dataset)
    corrects = 0

    disc.eval()
    with torch.no_grad():
        for inputs, gts in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()

            # Forward only
            _, outputs = disc(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == gts.data)

    acc = corrects / size
    print("Test Acc: {:.4f}".format(acc))
    return acc


g_losses = []
d_losses = []
best_acc_test = 0.0

for epoch in range(1, num_epochs + 1):
    gen.train()
    disc.train()

    avg_loss_g, avg_loss_d = 0.0, 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # train step
        loss_g, loss_d, acc_d = train_step(x, y)
        avg_loss_g += loss_g.detach().cpu().numpy() * x.shape[0]
        avg_loss_d += loss_d.detach().cpu().numpy() * x.shape[0]

        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: LossD: {:.6f} LossG: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_d, loss_g, acc_d))

    g_losses.append(avg_loss_g / len(train_dataset))
    d_losses.append(avg_loss_d / len(train_dataset))

    # Save
    if epoch % validate_every == 0:
        acc_test = test(test_loader)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            # Wrap things to a single dict to train multiple model weights
            state_dict = {
                "generator": gen.state_dict(),
                "discriminator": disc.state_dict(),
                }
            torch.save(state_dict, ckpt_path)
            print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))


        # Do some reconstruction
        gen.eval()
        with torch.no_grad():
            # Forward
            xg = gen(random_z)
            xg = denormalize(xg)

            # Plot 20 randomly generated images
            plt.figure(figsize=(10, 5))
            for p in range(20):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')

        # Plot losses
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim([1, epoch])
        plt.legend()
        plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)

# Just for you to check your Part 2 score
score = compute_score(best_acc_test, 0.65, 0.69)
print("Your final accuracy:", best_acc_test)
print("Your Assignment Score:", score)