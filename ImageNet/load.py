"""
File: load.py
Author: Chuncheng Zhang
Date: 2024-03-05
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Load the ImageNet images

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-05 ------------------------
# Requirements and constants
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet

import pandas as pd
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import display

root = Path(__file__).parent
data_path = root.joinpath('data')

# %% ---- 2024-03-05 ------------------------
# Function and class

transform = transforms.Compose([
    transforms.PILToTensor()
])

data_kwargs = dict(
    root=data_path,
    split='val',
)

load_kwargs = dict(
    batch_size=10,
    shuffle=True,
    num_workers=10
)

dataset = ImageNet(**data_kwargs)

# ! Not working yet
dataset_loader = DataLoader(
    dataset=dataset,
    **load_kwargs
)


class ImageNetDataSet(object):
    dataset = dataset

    def __init__(self):
        self.root = Path(self.dataset.root, self.dataset.split)
        self.search_folder()

    def search_folder(self):
        images = []
        xmls = []
        for e in tqdm(self.root.iterdir(), 'Searching...'):
            if e.is_dir():
                [images.append((e.name, f.name)) for f in e.iterdir()
                 if f.is_file() and f.suffix == '.JPEG']

            if e.suffix == '.xml':
                xmls.append((e.name,))

        images = pd.DataFrame(images, columns=['wnid', 'fileName'])
        images['cid'] = images['wnid'].map(
            lambda wnid: self.dataset.wnid_to_idx[wnid])
        xmls = pd.DataFrame(xmls, columns=['fileName'])

        def _fileName2id(name: str) -> int:
            return int(name.split('_')[-1].split('.')[0])

        def _better_df(df: pd.DataFrame) -> pd.DataFrame:
            df['uid'] = df['fileName'].map(_fileName2id)
            df.index = df['uid']
            df.index.name = None
            return df.sort_index()

        images = _better_df(images)
        xmls = _better_df(xmls)
        merge = xmls.merge(images, on='uid', suffixes=('.xml', '.jpeg'))

        display(merge)

        self.merge = merge
        self.images = images
        self.xmls = xmls
        self.n = len(images)

    def getitem(self, i: int = 0):
        i %= self.n
        select = self.merge.iloc[i]
        mass = self.wnid_to_mass(select['wnid'])

        # --------------------
        image = Image.open(
            Path(self.root, select['wnid'], select['fileName.jpeg']))

        # --------------------
        tree = ET.parse(Path(self.root, select['fileName.xml']))
        root = tree.getroot()
        xmls = []
        for obj in root.findall('object'):
            xml = dict(
                name=obj.find('name').text,
                pose=obj.find('pose').text,
                truncated=obj.find('truncated').text,
                difficult=obj.find('difficult').text,
                xmin=int(obj.find('bndbox').find('xmin').text),
                ymin=int(obj.find('bndbox').find('ymin').text),
                xmax=int(obj.find('bndbox').find('xmax').text),
                ymax=int(obj.find('bndbox').find('ymax').text),
            )
            xmls.append(xml)

        # --------------------
        mass.update(image=image, tree=tree, xmls=xmls, uid=select['uid'])
        return mass

    def class_names(self):
        return list(self.dataset.class_to_idx)

    def wnids(self):
        return list(self.dataset.wnid_to_idx)

    def class_to_mass(self, class_name: str) -> dict:
        cid = self.dataset.class_to_idx[class_name]
        wnid = self.dataset.wnids[cid]
        return self.better_mass(dict(class_name=class_name, wnid=wnid, cid=cid))

    def wnid_to_mass(self, wnid: str) -> dict:
        cid = self.dataset.wnid_to_idx[wnid]
        class_names = self.dataset.classes[cid]
        return self.better_mass(dict(class_name=class_names[0], wnid=wnid, cid=cid))

    def better_mass(self, mass: dict) -> dict:
        # u = self.dataset.class_to_idx[mass['class_name']]
        # classes = [k for k, v in self.dataset.class_to_idx.items() if v == u]
        classes = self.dataset.classes[mass['cid']]
        mass.update(classes=classes)
        return mass


# %% ---- 2024-03-05 ------------------------
# Play ground
if __name__ == '__main__':
    import random

    dataset = ImageNetDataSet()

    got = dataset.getitem(random.choice(range(50000000)))
    display(got)

    # --------------------
    image = got['image'].copy()
    draw = ImageDraw.Draw(image)
    font_path = Path(root, 'agave regular Nerd Font Complete Mono.ttf')
    font = ImageFont.truetype(font_path, 15)
    color = 'green'
    for i, xml in enumerate(got['xmls']):
        wnid = xml['name']
        text = '{} | {}'.format(dataset.wnid_to_mass(
            wnid)['classes'][0], dataset.wnid_to_mass(wnid)['cid'])
        draw.rectangle((xml['xmin'], xml['ymin'], xml['xmax'],
                       xml['ymax']), outline=color, width=2)
        left, top, right, bottom = draw.textbbox(
            (xml['xmin'], xml['ymin']-15), text=text, font=font, align='left')
        draw.rectangle((left, top-2, right+8, bottom), fill=color)
        draw.text((xml['xmin']+4, xml['ymin']-15),
                  text=text, font=font, align='left')
    display(image)

    # display(dataset.classes)
    # display(dataset.class_to_idx)
    # display(dataset.wnids)
    # display(dataset.wnid_to_idx)

# %%

# %%

# %% ---- 2024-03-05 ------------------------
# Pending


# %% ---- 2024-03-05 ------------------------
# Pending

# %%
