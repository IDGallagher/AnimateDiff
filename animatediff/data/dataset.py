import os, io, csv, math, random
import sys
import numpy as np
from einops import rearrange
from functools import partial
from decord import VideoReader
from torch.utils.data import DataLoader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print

sys.path.append("./webdataset/")
import webdataset as wds
import wids

def make_sample(sample, sample_size=256, sample_stride=4, sample_n_frames=16, is_image=False, **kwargs):
    video = sample[".mp4"]
    caption = sample[".txt"]

    print(f"sample size {sample_size}")
    
    video_reader = VideoReader(video)
    video_length = len(video_reader)
    
    if not is_image:
        clip_length = min(video_length, (sample_n_frames - 1) * sample_stride + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)
    else:
        batch_index = [random.randint(0, video_length - 1)]

    pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
    pixel_values = pixel_values / 255.
    del video_reader

    if is_image:
        pixel_values = pixel_values[0]

    sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
    pixel_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(sample_size[0]),
        transforms.CenterCrop(sample_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    return dict(pixel_values=pixel_transforms(pixel_values), text=caption)

def make_dataset(shards, cache_dir="./tmp", **kwargs):
    trainset = wids.ShardListDataset(shards, cache_dir=cache_dir, keep=True)
    trainset = trainset.add_transform(partial(make_sample, **kwargs))
    return trainset

def make_dataloader(dataset, batch_size=1):
    sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=2
    )
    return dataloader

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        try:
            zero_rank_print(f"loading annotations from {csv_path} ...")
            with open(csv_path, 'r') as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
            self.length = len(self.dataset)
            zero_rank_print(f"data scale: {self.length}")

            # zero_rank_print(f"data {self.dataset} ...")

            self.video_folder    = video_folder
            self.sample_stride   = sample_stride
            self.sample_n_frames = sample_n_frames
            self.is_image        = is_image
            
            sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
            self.pixel_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        except Exception as err:
            print(f"EXCEPTION {err}")
    
    def get_batch(self, idx):
        try:
            zero_rank_print(f"Get batch {idx} ...")
            # zero_rank_print(f"data {self.dataset} ...")
            video_dict = self.dataset[idx]
            videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
            zero_rank_print(f"folder {self.video_folder} {videoid}")
            video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
            zero_rank_print(f"Vid {video_dir}")
            video_reader = VideoReader(video_dir)
            video_length = len(video_reader)
            
            if not self.is_image:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx   = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            else:
                batch_index = [random.randint(0, video_length - 1)]

            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader

            if self.is_image:
                pixel_values = pixel_values[0]
        
        except Exception as err:
            print(f"EXCEPTION {err}")
            raise
        # zero_rank_print(f"Got batch {idx} ...")
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


