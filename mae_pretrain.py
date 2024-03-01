import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=1001)
    parser.add_argument('--warmup_epoch', type=int, default=40)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--gpu', type=str, default='1')

    parser.add_argument('--train_root', type=str, default='/home/aau3090/Datasets/celeba/train')
    parser.add_argument('--val_root', type=str, default='/home/aau3090/Datasets/celeba/val')
    #parser.add_argument('--train_root', type=str, default='/home/aau3090/Datasets/nozzle_dataset/spray_frames/generative/train')
    #parser.add_argument('--val_root', type=str, default='/home/aau3090/Datasets/nozzle_dataset/spray_frames/generative/val_sample')
    parser.add_argument('--folders', type=list, default=[])
    parser.add_argument('--n_workers', type=int, default=6)

    # model
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--encoder_emb_dim', type=int, default=192)
    parser.add_argument('--encoder_layer', type=int, default=12)
    parser.add_argument('--encoder_head', type=int, default=4)
    parser.add_argument('--decoder_emb_dim', type=int, default=512)
    parser.add_argument('--decoder_layer', type=int, default=8)
    parser.add_argument('--decoder_head', type=int, default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.75)

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    #train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    #val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    from dataset import ImageFolderDataset
    from transforms import train_transforms
    trainset = ImageFolderDataset(args.train_root, args.folders, train_transforms(frame_size=178, crop_size=args.image_size, mean=0.5, std=0.5, norm=True))
    valset = ImageFolderDataset(args.val_root, args.folders, train_transforms(frame_size=178, crop_size=args.image_size, mean=0.5, std=0.5, norm=True))
    #trainset = ImageFolderDataset(args.train_root, args.folders, train_transforms(frame_size=484, crop_size=args.image_size))
    #valset = ImageFolderDataset(args.val_root, args.folders, train_transforms(frame_size=484, crop_size=args.image_size))
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.n_workers)
    #val_dataloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.n_workers)

    writer = SummaryWriter(os.path.join('logs', 'face', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(image_size=args.image_size,
                    in_channel=args.in_channel,
                    patch_size=args.patch_size,
                    emb_dim=args.encoder_emb_dim,
                    encoder_layer=args.encoder_layer,
                    encoder_head=args.encoder_head,
                    decoder_layer=args.decoder_layer,
                    decoder_head=args.decoder_head,
                    mask_ratio=args.mask_ratio).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([valset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model, args.model_path)