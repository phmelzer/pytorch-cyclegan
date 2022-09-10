import argparse
import itertools
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
import xlsxwriter

from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import ReplayBuffer
from cyclegan_pytorch import weights_init

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--training_name", type=str, default="test",
                    help="name of training.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("--dataset", type=str, default="depthmaps_sim_grey_2000_real_1769",
                    help="dataset name. (default:`depthmaps_sim_grey_250_real_250`)")
parser.add_argument("--epochs", default=40, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--decay_epochs", type=int, default=39,
                    help="epoch to start linearly decaying the learning rate to 0. (default:100)")
parser.add_argument("-b", "--batch-size", default=1, type=int,
                    metavar="N")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate. (default:0.0002)")
parser.add_argument("--print_freq", default=100, type=int,
                    metavar="N", help="print frequency. (default:100)")
parser.add_argument("--cycle_loss_weight", type=int, default=10)
parser.add_argument("--identity_loss_weight", type=int, default=0)
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG_sim2real", default="", help="path to netG_A2B (to continue training)")
parser.add_argument("--netG_real2sim", default="", help="path to netG_B2A (to continue training)")
parser.add_argument("--netD_sim", default="", help="path to netD_sim (to continue training)")
parser.add_argument("--netD_real", default="", help="path to netD_real (to continue training)")
parser.add_argument("--image_size", type=int, default=128,
                    help="size of the data crop (squared assumed). (default:128)")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
# Dataset
dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                       transform=transforms.Compose([
                           transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                           transforms.RandomCrop(args.image_size),
                           transforms.ToTensor()]),
                       unaligned=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

try:
    os.makedirs(os.path.join(args.outf, args.training_name, "plots"))
    os.makedirs(os.path.join(args.outf, args.training_name, "images"))
    os.makedirs(os.path.join(args.outf, args.training_name, "weights"))
except OSError:
    pass


device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
netG_sim2real = Generator().to(device)
netG_real2sim = Generator().to(device)
netD_sim = Discriminator().to(device)
netD_real = Discriminator().to(device)

netG_sim2real.apply(weights_init)
netG_real2sim.apply(weights_init)
netD_sim.apply(weights_init)
netD_real.apply(weights_init)

if args.netG_sim2real != "":
    netG_sim2real.load_state_dict(torch.load(args.netG_sim2real))
if args.netG_real2sim != "":
    netG_real2sim.load_state_dict(torch.load(args.netG_real2sim))
if args.netD_sim != "":
    netD_sim.load_state_dict(torch.load(args.netD_sim))
if args.netD_real != "":
    netD_real.load_state_dict(torch.load(args.netD_real))

# define loss function (adversarial_loss) and optimizer
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_sim2real.parameters(), netG_real2sim.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_D_sim = torch.optim.Adam(netD_sim.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_real = torch.optim.Adam(netD_real.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D_sim = torch.optim.lr_scheduler.LambdaLR(optimizer_D_sim, lr_lambda=lr_lambda)
lr_scheduler_D_real = torch.optim.lr_scheduler.LambdaLR(optimizer_D_real, lr_lambda=lr_lambda)

epoch_list = []
index_list = []

loss_identity_real_list, loss_identity_sim_list = [],[]
loss_identity_rating_real_list, loss_identity_rating_sim_list = [],[]
loss_GAN_real2sim_list, loss_GAN_sim2real_list = [],[]
loss_cycle_realsimreal_list, loss_cycle_simrealsim_list = [],[]

fake_sim_buffer = ReplayBuffer()
fake_real_buffer = ReplayBuffer()

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        real_image_sim = data["Sim"].to(device)
        real_image_real = data["Real"].to(device)
        batch_size = real_image_sim.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Set G_A and G_B's gradients to zero
        optimizer_G.zero_grad()

        # Identity loss
        # G_real2sim(sim) should equal sim if real sim is fed
        identity_image_sim = netG_real2sim(real_image_sim)
        loss_identity_sim = identity_loss(identity_image_sim, real_image_sim) * args.identity_loss_weight
        # G_sim2real(real) should equal real if real real is fed
        identity_image_real = netG_sim2real(real_image_real)
        loss_identity_real = identity_loss(identity_image_real, real_image_real) * args.identity_loss_weight

        # GAN loss
        # GAN loss D_sim(G_sim(sim))
        fake_image_sim = netG_real2sim(real_image_real)
        fake_output_sim = netD_sim(fake_image_sim)
        loss_GAN_real2sim = adversarial_loss(fake_output_sim, real_label)
        # GAN loss D_real(G_real(real))
        fake_image_real = netG_sim2real(real_image_sim)
        fake_output_real = netD_real(fake_image_real)
        loss_GAN_sim2real = adversarial_loss(fake_output_real, real_label)

        # Cycle loss
        recovered_image_sim = netG_real2sim(fake_image_real)
        loss_cycle_simrealsim = cycle_loss(recovered_image_sim, real_image_sim) * args.cycle_loss_weight

        recovered_image_real = netG_sim2real(fake_image_sim)
        loss_cycle_realsimreal = cycle_loss(recovered_image_real, real_image_real) * args.cycle_loss_weight

        # identity rating loss
        loss_identity_rating_sim = identity_loss(real_image_sim, fake_image_sim)
        loss_identity_rating_real = identity_loss(real_image_real, fake_image_real)

        # Combined loss and calculate gradients
        errG = loss_identity_sim + loss_identity_real + loss_GAN_sim2real + loss_GAN_real2sim + loss_cycle_simrealsim + loss_cycle_realsimreal

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        optimizer_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        optimizer_D_sim.zero_grad()

        # Real A image loss
        real_output_A = netD_sim(real_image_sim)
        errD_real_sim = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_image_sim = fake_sim_buffer.push_and_pop(fake_image_sim)
        fake_output_sim = netD_sim(fake_image_sim.detach())
        errD_fake_sim = adversarial_loss(fake_output_sim, fake_label)

        # Combined loss and calculate gradients
        errD_sim = (errD_real_sim + errD_fake_sim) / 2

        # Calculate gradients for D_A
        errD_sim.backward()
        # Update D_A weights
        optimizer_D_sim.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        optimizer_D_real.zero_grad()

        # Real B image loss
        real_output_B = netD_real(real_image_real)
        errD_real_real = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_real = fake_real_buffer.push_and_pop(fake_image_real)
        fake_output_real = netD_real(fake_image_real.detach())
        errD_fake_real = adversarial_loss(fake_output_real, fake_label)

        # Combined loss and calculate gradients
        errD_real = (errD_real_real + errD_fake_real) / 2

        # Calculate gradients for D_B
        errD_real.backward()
        # Update D_B weights
        optimizer_D_real.step()

        progress_bar.set_description(
            f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(errD_sim + errD_real).item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"Loss_G_identity: {(loss_identity_sim + loss_identity_real).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_sim2real + loss_GAN_real2sim).item():.4f} "
            f"loss_G_cycle: {(loss_cycle_simrealsim + loss_cycle_realsimreal).item():.4f} "
            f"loss_identity_rating_real: {loss_identity_rating_real.item():.4f} "
            f"loss_identity_rating_sim: {loss_identity_rating_sim.item():.4f}"
        )

        if i % args.print_freq == 0:
            vutils.save_image(real_image_sim,
                              f"{args.outf}/{args.training_name}/images/real_image_sim.png",
                              normalize=True)
            vutils.save_image(real_image_real,
                              f"{args.outf}/{args.training_name}/images/real_image_real.png",
                              normalize=True)

            fake_image_sim = 0.5 * (netG_real2sim(real_image_real).data + 1.0)
            fake_image_real = 0.5 * (netG_sim2real(real_image_sim).data + 1.0)


            vutils.save_image(fake_image_sim.detach(),
                              f"{args.outf}/{args.training_name}/images/fake_image_sim.png",
                              normalize=True)
            vutils.save_image(fake_image_real.detach(),
                              f"{args.outf}/{args.training_name}/images/fake_image_real.png",
                              normalize=True)
            
            transformed_image_sim = 0.5 * (netG_real2sim(fake_image_real).data + 1.0)
            transformed_image_real = 0.5 * (netG_sim2real(fake_image_sim).data + 1.0)


            vutils.save_image(transformed_image_sim.detach(),
                              f"{args.outf}/{args.training_name}/images/transformed_image_sim.png",
                              normalize=True)
            vutils.save_image(transformed_image_real.detach(),
                              f"{args.outf}/{args.training_name}/images/transformed_image_real.png",
                              normalize=True)

            # merge results
            new_img = Image.new('RGB', (args.image_size*3,args.image_size*2),'white')
            real_sample_sim = Image.open(f"{args.outf}/{args.training_name}/images/real_image_sim.png")
            fake_sample_real = Image.open(f"{args.outf}/{args.training_name}/images/fake_image_real.png")
            transformed_image_sim = Image.open(f"{args.outf}/{args.training_name}/images/transformed_image_sim.png")
            transformed_image_real = Image.open(f"{args.outf}/{args.training_name}/images/transformed_image_real.png")
            real_sample_real = Image.open(f"{args.outf}/{args.training_name}/images/real_image_real.png")
            fake_sample_sim = Image.open(f"{args.outf}/{args.training_name}/images/fake_image_sim.png")
            new_img.paste(real_sample_sim,(0,0,args.image_size,args.image_size))
            new_img.paste(fake_sample_real,(args.image_size,0,args.image_size*2,args.image_size))
            new_img.paste(transformed_image_sim,(args.image_size*2,0,args.image_size*3,args.image_size))
            new_img.paste(real_sample_real,(0,args.image_size,args.image_size,args.image_size*2))
            new_img.paste(fake_sample_sim,(args.image_size,args.image_size,args.image_size*2,args.image_size*2))
            new_img.paste(transformed_image_real,(args.image_size*2,args.image_size,args.image_size*3,args.image_size*2))
            new_img.save(f"{args.outf}/{args.training_name}/images/results_at_epoch_{epoch}_idx_{i}.png")

            epoch_list.append(epoch)
            index_list.append(i)
            loss_identity_sim_list.append(loss_identity_sim.item())
            loss_identity_real_list.append(loss_identity_real.item())
            loss_GAN_real2sim_list.append(loss_GAN_real2sim.item())
            loss_GAN_sim2real_list.append(loss_GAN_sim2real.item())
            loss_cycle_simrealsim_list.append(loss_cycle_simrealsim.item())
            loss_cycle_realsimreal_list.append(loss_cycle_realsimreal.item())
            loss_identity_rating_sim_list.append(loss_identity_rating_sim.item())
            loss_identity_rating_real_list.append(loss_identity_rating_real.item())

    # do check pointing
    torch.save(netG_sim2real.state_dict(), f"{args.outf}/{args.training_name}/weights/netG_sim2real_epoch_{epoch}.pth")
    torch.save(netG_real2sim.state_dict(), f"{args.outf}/{args.training_name}/weights/netG_real2sim_epoch_{epoch}.pth")
    torch.save(netD_sim.state_dict(), f"{args.outf}/{args.training_name}/weights/netD_sim_epoch_{epoch}.pth")
    torch.save(netD_real.state_dict(), f"{args.outf}/{args.training_name}/weights/netD_real_epoch_{epoch}.pth")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_sim.step()
    lr_scheduler_D_real.step()

# save last check pointing
torch.save(netG_sim2real.state_dict(), f"{args.outf}/{args.training_name}/weights/netG_sim2real.pth")
torch.save(netG_real2sim.state_dict(), f"{args.outf}/{args.training_name}/weights/netG_real2sim.pth")
torch.save(netD_sim.state_dict(), f"{args.outf}/{args.training_name}/weights/netD_sim.pth")
torch.save(netD_real.state_dict(), f"{args.outf}/{args.training_name}/weights/netD_real.pth")

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_GAN_real2sim_list))
plt.title('Loss GAN Real2Sim')
plt.ylabel('Loss GAN Real2Sim')
plt.ylim(0,2)
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_gan_real2sim.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_GAN_sim2real_list))
plt.title('Loss GAN Sim2Real')
plt.ylabel('Loss GAN Sim2Real')
plt.ylim(0,2)
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_gan_sim2real.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_cycle_simrealsim_list))
plt.title('Loss Cycle SimRealSim')
plt.ylabel('Loss Cycle SimRealSim')
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_cycle_simrealsim.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_cycle_realsimreal_list))
plt.title('Loss Cycle RealSimReal')
plt.ylabel('Loss Cycle RealSimReal')
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}//{args.training_name}/plots/loss_cycle_realsimreal.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_identity_rating_sim_list))
plt.title('Loss Identity Rating Sim')
plt.ylabel('Loss Identity Rating Sim')
plt.ylim(0,1)
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_identity_rating_sim.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_identity_rating_real_list))
plt.title('Loss Identity Rating Real')
plt.ylabel('Loss Identity Rating Real')
plt.ylim(0,1)
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_identity_rating_real.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_identity_sim_list))
plt.title('Loss Identity Sim')
plt.ylabel('Loss Identity Sim')
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_identity_sim.png")
# plt.show()
plt.close()

window = 10
plt.figure(figsize=(12, 8))
plt.plot(np.array(loss_identity_real_list))
plt.title('Loss Identity Real')
plt.ylabel('Loss Identity Real')
plt.xlabel('Iterations (5 Iterations = 1 Epoch)')
plt.legend(loc="lower right")
plt.savefig(f"{args.outf}/{args.training_name}/plots/loss_identity_real.png")
# plt.show()
plt.close()

data = xlsxwriter.Workbook(f"{args.outf}/{args.training_name}/data.xlsx")
worksheet = data.add_worksheet('configurations')

worksheet.write('A1', 'training_name')
worksheet.write('A2', 'dataroot')
worksheet.write('A3', 'dataset')
worksheet.write('A4', 'epochs')
worksheet.write('A5', 'decay_epochs')
worksheet.write('A6', 'batch_size')
worksheet.write('A7', 'lr')
worksheet.write('A8', 'print_freq')
worksheet.write('A9', 'cycle_loss_weight')
worksheet.write('A10', 'identity_loss_weight')
worksheet.write('A11', 'image_size')

worksheet.write('B1', args.training_name)
worksheet.write('B2', args.dataroot)
worksheet.write('B3', args.dataset)
worksheet.write('B4', args.epochs)
worksheet.write('B5', args.decay_epochs)
worksheet.write('B6', args.batch_size)
worksheet.write('B7', args.lr)
worksheet.write('B8', args.print_freq)
worksheet.write('B9', args.cycle_loss_weight)
worksheet.write('B10', args.identity_loss_weight)
worksheet.write('B11', args.image_size)

worksheet1 = data.add_worksheet('loss functions')

worksheet1.write('A1', 'Epoch')
worksheet1.write('B1', 'Index')
worksheet1.write('C1', 'Loss GAN Real2Sim')
worksheet1.write('D1', 'Loss GAN Sim2Real')
worksheet1.write('E1', 'Loss Cycle SimRealSim')
worksheet1.write('F1', 'Loss Cycle RealSimReal')
worksheet1.write('G1', 'Loss Identity Rating Sim')
worksheet1.write('H1', 'Loss Identity Rating Real')
worksheet1.write('I1', 'Loss Identity Sim')
worksheet1.write('J1', 'Loss Identity Real')

rowindex = 2

for i in range(len(loss_GAN_real2sim_list)):
    worksheet1.write('A' + str(rowindex), epoch_list[i])
    worksheet1.write('B' + str(rowindex), index_list[i])
    worksheet1.write('C' + str(rowindex), loss_GAN_real2sim_list[i])
    worksheet1.write('D' + str(rowindex), loss_GAN_sim2real_list[i])
    worksheet1.write('E' + str(rowindex), loss_cycle_simrealsim_list[i])
    worksheet1.write('F' + str(rowindex), loss_cycle_realsimreal_list[i])
    worksheet1.write('G' + str(rowindex), loss_identity_rating_sim_list[i])
    worksheet1.write('H' + str(rowindex), loss_identity_rating_real_list[i])
    worksheet1.write('I' + str(rowindex), loss_identity_sim_list[i])
    worksheet1.write('J' + str(rowindex), loss_identity_real_list[i])

    rowindex += 1

data.close()

