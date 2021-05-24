from __future__ import print_function
import streamlit as st


from src.misc.config import cfg, cfg_from_file
from src.dataset import TextDataset
from src.trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a AttnGAN network')
#     parser.add_argument('--cfg', dest='cfg_file',
#                         help='optional config file',
#                         default='cfg/bird_attn2.yml', type=str)
#     parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
#     parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
#     parser.add_argument('--manualSeed', type=int, help='manual seed')
#     args = parser.parse_args()
#     return args


def gen_example(wordtoix, algo, text):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    data_dic = {}

    captions = []
    cap_lens = []
    
    sent = text.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
   

    rev = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    name = "input"
    key = name[(name.rfind('/') + 1):]
    data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    # args = parse_args()
    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file)

    # if args.gpu_id != -1:
    #     cfg.GPU_ID = args.gpu_id
    # else:
    #     cfg.CUDA = False

    # if args.data_dir != '':
    #     cfg.DATA_DIR = args.data_dir
    cfg = "eval_bird.yml"
    cfg_from_file(cfg)
    print('Using config:')
    pprint.pprint(cfg)

    # if not cfg.TRAIN.FLAG:
    #     args.manualSeed = 100
    # elif args.manualSeed is None:
    #     args.manualSeed = random.randint(1, 10000)
    # random.seed(args.manualSeed)
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)
    # if cfg.CUDA:
    #     torch.cuda.manual_seed_all(args.manualSeed)

    # now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # output_dir = '../output/%s_%s_%s' % \
        # (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    # split_dir, bshuffle = 'train', True
    # if not cfg.TRAIN.FLAG:
    #     # bshuffle = False
    #     split_dir = 'test'

    # Get data loader
    output_dir = "output/"
    split_dir = "test"
    bshuffle = True
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()     
    gen_example(dataset.wordtoix, algo, text="A black bird")  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
