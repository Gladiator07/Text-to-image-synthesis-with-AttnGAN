from __future__ import print_function

from src.misc.config import cfg, cfg_from_file
from src.dataset import TextDataset
from src.trainer import condGANTrainer as trainer

import time
import random
import pprint
import numpy as np

import torch
import torchvision.transforms as transforms


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
    name = "output"
    key = name[(name.rfind('/') + 1):]
    data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":

    import streamlit as st

    st.header("Text To Image Synthesis using AttnGAN")

    user_input = st.text_input("Enter the bird description")

    cfg_from_file('eval_bird.yml')
    print('Using config:')
    pprint.pprint(cfg)
    cfg.CUDA = False
    manualSeed = 100
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
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
    if user_input:

        start_t = time.time()
        # generate images for customized captions
        gen_example(dataset.wordtoix, algo,
                    text=user_input)
        end_t = time.time()
        print('Total time for training:', end_t - start_t)

        st.image("models/bird_AttnGAN2/output/0_s_0_g2.png")

        st.write("The attention given for each word")
        st.image("models/bird_AttnGAN2/output/0_s_0_a1.png")

        # with st.section(label="show the first stage created images"):
        #     st.image("models/bird_AttnGAN2/output/0_s_0_a0.png")

        with st.beta_expander("click to see the first stage image"):
            st.write("First stage image")
            st.image("models/bird_AttnGAN2/output/0_s_0_g1.png")
            st.write("First stage attention on image")
            st.image("models/bird_AttnGAN2/output/0_s_0_a0.png")

