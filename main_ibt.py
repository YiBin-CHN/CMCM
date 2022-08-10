#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import numpy as np
from data_loader import get_loader
import random
import argparse
import time
from model_ibt import train, decode
from pathlib import Path
import json
from torchvision import transforms
from PIL import Image
import pickle
from build_vocab import Vocabulary
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

#glove_file = datapath('.../glove.42B.300d.txt')#glove to word2vec
tmp_file = get_tmpfile('.../test_word2vec.txt')
#from gensim.scripts.glove2word2vec import glove2word2vec#
#glove2word2vec(glove_file, tmp_file)#

def parse_args():
    parser = argparse.ArgumentParser(description='order')
    parser.add_argument('--model', type=str, default='imgbytx')
    parser.add_argument('--main_path', type=str, default="./")
    parser.add_argument('--model_path', type=str, default="models")
    parser.add_argument('--decoding_path', type=str, default="decoding")
    parser.add_argument('--writetrans', type=str, default='./decoding/img_order.devorder',
                       help='write translations for to a file')
    parser.add_argument('--ref', type=str, help='references, word unit')

    parser.add_argument('--vocab_path', type=str, default='./voc/vocabfil5.pkl',
                        help='path for vocabulary wrapper')

    #path of data
    parser.add_argument('--train_image_dir', type=str, default='.../train',
                        help='directory for resized train images')
    parser.add_argument('--val_image_dir', type=str, default='.../val',
                        help='directory for resized val images')
    parser.add_argument('--train_sis_path', type=str,
                        default='.../train.story-in-sequence.json',
                        help='path for train sis json file')
    parser.add_argument('--val_sis_path', type=str,
                        default='.../val.story-in-sequence.json',
                        help='path for val sis json file')
    parser.add_argument('--test_image_dir', type=str, default='.../test',
                        help='directory for resized test images')
    parser.add_argument('--test_sis_path', type=str,
                        default='.../test.story-in-sequence.json')


    parser.add_argument('--image_size', type=int, default=224, help='size for input images')
    parser.add_argument('--img_feature_size', type=int, default=512,
                        help='dimension of image feature')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--d_rnn', type=int, default=512, help='hidden dimention size')
    parser.add_argument('--d_mlp', type=int, default=512, help='dimention size for FFN')
    parser.add_argument('--gnnl', default=2, type=int, help='stacked layer number')
    parser.add_argument('--attdp', default=0.1, type=float, help='self-att dropout')

    parser.add_argument('--initnn', default='standard', help='parameter init')
    parser.add_argument('--early_stop', type=int, default=3)

    # running setting
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')

    parser.add_argument('--keep_cpts', type=int, default=1, help='save n checkpoints, when 1 save best model only')

    # training
    parser.add_argument('--eval_every', type=int, default=100, help='validate every * step')
    parser.add_argument('--save_every', type=int, default=2, help='save model every * step (5000)')
    parser.add_argument('--delay', type=int, default=1)

    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--margin', type=float, default=0.2, help='loss2')
    parser.add_argument('--lamb', type=float, default=1, help='lamb for loss')

    # lr decay
    parser.add_argument('--lrdecay', type=float, default=0, help='learning rate decay')
    parser.add_argument('--patience', type=int, default=0, help='learning rate decay 0.5')

    parser.add_argument('--maximum_steps', type=int, default=100, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--input_drop_ratio', type=float, default=0.1, help='dropout ratio only for inputs')

    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping')

    # model saving/reloading, output translations
    parser.add_argument('--load_from',default='', help='load from 1.modelname, 2.lastnumber, 3.number')

    parser.add_argument('--resume', action='store_true',
                        help='when resume, need other things besides parameters')

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        if args.load_from is not None and len(args.load_from) == 1:
            load_from = args.load_from[0]
            print('{} load the checkpoint from {} for initilize or resume'.
                  format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            checkpoint = None

        # if not resume(initilize), only need model parameters
        if args.resume:
            print('update args from checkpoint')
            load_dict = checkpoint['args'].__dict__
            except_name = ['mode', 'resume', 'maximum_steps']
            override(args, load_dict, tuple(except_name))

        main_path = Path(args.main_path)
        model_path = main_path / args.model_path
        decoding_path = main_path / args.decoding_path

        for path in [model_path, decoding_path]:
            path.mkdir(parents=True, exist_ok=True)

        args.model_path = str(model_path)
        args.decoding_path = str(decoding_path)

        if args.model == '[time]':
            args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

        # setup random seeds
        set_seeds(args.seed)

        # Image preprocessing
        train_transform = transforms.Compose([
            transforms.RandomCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        val_transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        test_transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        # Load vocabulary wrapper.
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        args.__dict__.update({'doc_vocab': len(vocab) + 1})

        wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        vocab_size = len(vocab) + 1
        embed_size = 300
        weight = torch.zeros(vocab_size + 1, embed_size)

        for i in range(len(wvmodel.index_to_key)):#index2word index_to_key
            try:
                index = vocab.word2idx[wvmodel.index_to_key[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(
                vocab.idx2word[vocab.word2idx[wvmodel.index_to_key[i]]]))

        # Build data loader
        train_data_loader = get_loader(args.train_image_dir, args.train_sis_path, vocab, train_transform,
                                       args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_data_loader = get_loader(args.val_image_dir, args.val_sis_path, vocab, val_transform, 1,
                                     shuffle=False, num_workers=args.num_workers)
        test_data_loader = get_loader(args.test_image_dir, args.test_sis_path, vocab, test_transform, 1,
                                     shuffle=False, num_workers=args.num_workers)
        print('{} Start training'.format(curtime()))
        train(args, train_data_loader, val_data_loader, test_data_loader, weight, checkpoint)

    else:
        load_from = '{}.best.pt'.format(args.load_from)
        print('{} load the best checkpoint from {}'.format(curtime(), load_from))
        checkpoint = torch.load(load_from, map_location='cpu')
        # when translate load_dict update args except some
        print('update args from checkpoint')
        load_dict = checkpoint['args'].__dict__

        print('{} Load test set'.format(curtime()))
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        args.__dict__.update({'doc_vocab': len(vocab)+2})
        test_transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        test_data_loader = get_loader(args.test_image_dir, args.test_sis_path, vocab, test_transform, 1,
                                      shuffle=False, num_workers=args.num_workers)
        start = time.time()
        decode(args, test_data_loader, checkpoint)
        print('{} Decode done, time {} mins'.format(curtime(), (time.time() - start) / 60))
