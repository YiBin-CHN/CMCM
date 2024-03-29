import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from vist import VIST


class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path, vocab, transform=None):
        self.image_dir = image_dir
        self.vist = VIST(sis_path)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        sentence = []
        images = []
        order = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            order.append(annotation["worker_arranged_photo_order"])

            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.extend([vocab(token) for token in tokens])
            target = torch.Tensor(caption)
            sentence.append(target)

        return torch.stack(images), sentence, order


    def __len__(self):
        return len(self.ids)

    def GetItem(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        sentence = []
        images = []
        order = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            order.append(annotation["worker_arranged_photo_order"])

            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, image_id + image_format)).convert('RGB')
                    break
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.extend([vocab(token) for token in tokens])
            target = torch.Tensor(caption)
            sentence.append(target)

        return images, sentence, order

    def GetLength(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, caption_stories, order_set = zip(*data)

    captions_set = []
    lengths_set = []
    doc_num_set = []
    tgt_len_set = []

    for captions in caption_stories:
        length = [len(cap) for cap in captions]
        doc_n = [len(captions)]
        tgt_len = torch.tensor([len(captions)])
        lengths = torch.tensor(length)
        targets = torch.zeros(len(captions), max(length)).long()
        for i, cap in enumerate(captions):
            end = length[i]
            targets[i, :end] = cap[:end]

        captions_set.append(targets)
        lengths_set.append(lengths)
        tgt_len_set.append(tgt_len)
        doc_num_set.append(doc_n)


    return image_stories, captions_set, lengths_set, order_set, tgt_len_set, doc_num_set


def get_loader(root, sis_path, vocab, transform, batch_size, shuffle, num_workers):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
