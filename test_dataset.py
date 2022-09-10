import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image

from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("--dataset", type=str, default="depthmaps_validierung_sim_grey_100",
                    help="dataset name. (default:`depthmaps_validierung_sim_grey_100`)")
parser.add_argument("--dataset_weights", type=str, default="depthmaps_sim_grey_flip_500_real_flip_500",
                    help="dataset name. (default:`depthmaps_sim_grey_flip_500_real_flip_500`)")
parser.add_argument("--weights_sim2real", type=str, default="netG_sim2real_epoch_38.pth",
                    help="dataset name. (default:`horse2zebra`)")
parser.add_argument("--weights_real2sim", type=str, default="netG_real2sim_epoch_38.pth",
                    help="dataset name. (default:`horse2zebra`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--outf", default="./results_validation",
                    help="folder to output images. (default: `./results`).")
parser.add_argument("--image-size", type=int, default=128,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 99)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset
dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                       transform=transforms.Compose([
                           transforms.Resize(args.image_size),
                           transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

try:
    os.makedirs(os.path.join(args.outf, str(args.dataset)))
    os.makedirs(os.path.join(args.outf, str(args.dataset)))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
netG_sim2real = Generator().to(device)
netG_real2sim = Generator().to(device)

# Load state dicts
netG_sim2real.load_state_dict(torch.load(os.path.join("weights/", str(args.dataset_weights), str(args.weights_sim2real))))
netG_real2sim.load_state_dict(torch.load(os.path.join("weights/", str(args.dataset_weights), str(args.weights_real2sim))))

# Set model mode
netG_sim2real.eval()
netG_real2sim.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, data in progress_bar:
    # get batch size data
    real_image_sim = data["Sim"].to(device)
    real_image_real = data["Real"].to(device)

    # Generate output
    fake_image_sim = 0.5 * (netG_real2sim(real_image_real).data + 1.0)
    fake_image_real = 0.5 * (netG_sim2real(real_image_sim).data + 1.0)

    # Save image files
    vutils.save_image(real_image_sim,
                      f"{args.outf}/{args.dataset}/real_sample_sim.png",
                      normalize=True)
    vutils.save_image(real_image_real,
                      f"{args.outf}/{args.dataset}/real_sample_real.png",
                      normalize=True)

    vutils.save_image(fake_image_sim.detach(),
                      f"{args.outf}/{args.dataset}/fake_sample_sim.png",
                      normalize=True)
    vutils.save_image(fake_image_real.detach(),
                      f"{args.outf}/{args.dataset}/fake_sample_real.png",
                      normalize=True)

    # merge results
    new_img = Image.new('RGB', (args.image_size * 2, args.image_size * 2), 'white')
    real_sample_sim = Image.open(f"{args.outf}/{args.dataset}/real_sample_sim.png")
    fake_sample_real = Image.open(f"{args.outf}/{args.dataset}/fake_sample_real.png")
    real_sample_real = Image.open(f"{args.outf}/{args.dataset}/real_sample_real.png")
    fake_sample_sim = Image.open(f"{args.outf}/{args.dataset}/fake_sample_sim.png")
    new_img.paste(real_sample_sim, (0, 0, args.image_size, args.image_size))
    new_img.paste(fake_sample_real, (args.image_size, 0, args.image_size * 2, args.image_size))
    new_img.paste(real_sample_real, (0, args.image_size, args.image_size, args.image_size * 2))
    new_img.paste(fake_sample_sim, (args.image_size, args.image_size, args.image_size * 2, args.image_size * 2))
    new_img.save(f"{args.outf}/{args.dataset}/results_at_{i}.png")

    progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")
