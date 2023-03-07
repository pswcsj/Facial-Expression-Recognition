import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np

__all__ = ['FERTrainDataLoader', 'FERTrainDataSet', 'FERTestDataSet']


class FERTrainDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=0):
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45, 45)),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor()
        ])

        self.dataset = FERTrainDataSet(transform=trsfm)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class FERTestDataLoader(DataLoader):
    def __init__(self, batch_size=1, shuffle=False, num_workers=0):
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.dataset = FERTestDataSet(transform=trsfm)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class FERTrainDataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # 캐시에 저장되어 있는지 확인하여 만약 저장되어 있다면 저장된 값을 사용
        if os.path.isfile(dir_path + '/cache/train/data.pt') and os.path.isfile(dir_path + '/cache/train/label.pt'):
            self.emotion = torch.load(dir_path + '/cache/train/label.pt')
            self.data = torch.load(dir_path + '/cache/train/data.pt')
            self.len = self.emotion.shape[0]
            return

        # csv 파일로부터 데이터 받아오는 부분
        df = pd.read_csv('dataset/fer2013.csv')
        df = df[df['Usage'] == 'Training']

        self.emotion = torch.LongTensor(df['emotion'].values)
        self.data = df['pixels'].apply(
            lambda a: torch.FloatTensor(list(map(int, a.split(' ')))).reshape(1, 48, 48)).values
        self.len = self.emotion.shape[0]

        torch.save(self.data, 'data_loader/cache/train/data.pt')
        torch.save(self.emotion, 'data_loader/cache/train/label.pt')

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.emotion[index]
        return self.data[index], self.emotion[index]

    def __len__(self):
        return self.len


class FERTestDataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        dir_path = os.path.dirname(os.path.realpath(__file__))

        if os.path.isfile(dir_path + '/cache/test/data.pt') and os.path.isfile(dir_path + '/cache/test/label.pt'):
            self.emotion = torch.load(dir_path + '/cache/test/label.pt')
            self.data = torch.load(dir_path + '/cache/test/data.pt')
            self.len = self.emotion.shape[0]
            return

        df = pd.read_csv('dataset/fer2013.csv')
        df = df[(df['Usage'] == 'PrivateTest') | (df['Usage'] == 'PublicTest')]
        self.emotion = torch.LongTensor(df['emotion'].values)
        self.data = df['pixels'].apply(
            lambda a: torch.FloatTensor(list(map(int, a.split(' ')))).reshape(1, 48, 48)).values
        self.len = self.emotion.shape[0]

        torch.save(self.data, 'data_loader/cache/test/data.pt')
        torch.save(self.emotion, 'data_loader/cache/test/label.pt')

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.emotion[index]
        return self.data[index], self.emotion[index]

    def __len__(self):
        return self.len


class AffectNetDataLoader(DataLoader):
    def __init__(self, path, batch_size, train=True, shuffle=True, num_workers=0):
        if train:
            trsfm = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
            # trsfm = transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation((-45, 45)),
            #     transforms.ColorJitter(brightness=0.5),
            #     transforms.ToTensor()
            # ])
        else:
            trsfm = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )

        self.dataset = AffectNetDataset(path, train=train, transform=trsfm)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class AffectNetDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.transform = transform
        self.path = path

        if train:
            self.img_path = self.path + '/train_set/images'
            self.label_path = self.path + '/train_set/annotations'
        else:
            self.img_path = self.path + '/val_set/images'
            self.label_path = self.path + '/val_set/annotations'

        self.img_list = []
        for img_path in filter(lambda u: u[0] != '.', os.listdir(self.img_path)):
            self.img_list.append(self.img_path+'/'+img_path)
        self.label_list = list(filter(lambda u: u[0] != '.', os.listdir(self.label_path)))
        self.len = len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        label = np.load(f'{self.label_path}/{self.label_list[index]}').astype(np.float)

        if self.transform:
            return self.transform(img), label
        return img, label

    def __len__(self):
        return self.len
