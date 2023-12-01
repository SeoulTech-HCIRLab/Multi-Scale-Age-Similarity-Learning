import random
from DataSelector import data_select
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as fn
from PIL import Image

IM_WIDTH = IM_HEIGHT = 224


class Loader(Dataset):

    def __init__(self, dataframe, datapath, transform=None, phase="test"):
        self.filenames = dataframe['filename'].values
        self.ages = dataframe['age'].values
        self.index = dataframe.index.values
        self.datapath = datapath
        self.transform = transform
        self.phase = phase
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, item):
        input = Image.open(self.datapath + self.filenames[item], mode='r')
        input_age = torch.tensor(self.ages[item], dtype=torch.float32)

        if self.transform is not None:
            input = self.transform_image(input)

        if self.phase == "train":
            positive_index = random.choice(
                self.index[self.index != item][self.ages[self.index != item] == self.ages[item]])
            negative_index = random.choice(
                self.index[self.index != item][self.ages[self.index != item] != self.ages[item]])

            positive = Image.open(self.datapath + self.filenames[positive_index], mode='r')
            negative = Image.open(self.datapath + self.filenames[negative_index], mode='r')

            if self.transform is not None:
                positive = self.transform_image(positive)
                negative = self.transform_image(negative)

            return input, positive, negative, input_age

        else:
            return input, input_age

    def transform_image(self, image):
        image = self.transform(image)
        image = fn.to_tensor(image)
        c, _, _ = image.size()
        if c == 1:
            image = torch.cat([image, image, image], dim=0)
        image = self.normalize(image)

        return image

    def __len__(self):
        return self.filenames.shape[0]


def get_transforms(phase):
    if phase == "train":
        transform = transforms.Compose([transforms.Resize((IM_HEIGHT, IM_WIDTH)),
                                        transforms.RandomGrayscale(0.1),
                                        transforms.RandomHorizontalFlip(0.2),
                                        transforms.RandomRotation(degrees=30),
                                        transforms.RandomAdjustSharpness(0.2),
                                        transforms.RandomVerticalFlip(0.2)])
        return transform

    elif phase == "test" or phase == "val":
        transform = transforms.Compose([transforms.Resize((IM_HEIGHT, IM_WIDTH))])
        return transform


def get_loader(phase, args):
    df = data_select(args.data_type, phase)
    transform = get_transforms(phase)
    if phase == "train":
        shuffle = True
    else:
        shuffle = False
    dataset = Loader(dataframe=df, datapath=args.data_path, transform=transform, phase=phase)
    loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=shuffle, num_workers=4)

    return loader
