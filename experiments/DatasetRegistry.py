import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os

class OCRImageDatasetV1(torch.utils.data.Dataset):
    def __init__(self, image_folder_root, labels_path):
        super().__init__()
        # image root folder
        self.image_folder_root = image_folder_root

        # labels dataframe (remove digits which are not of 12 digits)
        df = pd.read_csv(labels_path)
        df['label'] = df['label'].astype(str)
        self.labels_df = df[df['label'].str.len() == 12].reset_index(drop=True)
        print(f"Original size: {len(df)}, Filtered size: {len(self.labels_df)}")

        # transformation for data augmentation & image normalization 
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        # character mapping to digit
        char_set = "0123456789"
        self.char_map = {char:i for i, char in enumerate(char_set)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        image_i = Image.open(os.path.join(self.image_folder_root, self.labels_df['image_path'][idx]))
        image_i = self.transform(image_i)

        image_i = image_i.ravel()

        label_i = [self.char_map[char] for char in str(self.labels_df['label'][idx])]
        label_i = torch.tensor(label_i, dtype=torch.long)
        return image_i, label_i
    
class OCRImageDatasetV2(torch.utils.data.Dataset):
    IMAGE_HEIGHT = 0
    IMAGE_WIDTH = 0

    def __init__(self, image_folder_root, labels_path):
        super().__init__()
        # image root folder
        self.image_folder_root = image_folder_root

        # labels dataframe (remove digits which are not of 12 digits)
        df = pd.read_csv(labels_path)
        df['label'] = df['label'].astype(str)
        self.labels_df = df[df['label'].str.len() == 12].reset_index(drop=True)
        print(f"Original size: {len(df)}, Filtered size: {len(self.labels_df)}")

        # constant resizing parameters
        self.IMAGE_HEIGHT = 32
        self.IMAGE_WIDTH = 256

        # transformation for data augmentation & image normalization 
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([self.IMAGE_HEIGHT, self.IMAGE_WIDTH]),
            transforms.ToTensor()
        ])

        # character mapping to digit
        char_set = "0123456789"
        self.char_map = {char:i for i, char in enumerate(char_set)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        image_i = Image.open(os.path.join(self.image_folder_root, self.labels_df['image_path'][idx]))
        image_i = self.transform(image_i)

        #image_i = image_i.ravel()

        label_i = [self.char_map[char] for char in str(self.labels_df['label'][idx])]
        label_i = torch.tensor(label_i, dtype=torch.long)
        return image_i, label_i
    
class OCRImageDatasetV4(torch.utils.data.Dataset):
    IMAGE_HEIGHT = 0
    IMAGE_WIDTH = 0

    def __init__(self, image_folder_root, labels_path):
        super().__init__()
        # image root folder
        self.image_folder_root = image_folder_root

        # labels dataframe (remove digits which are not of 12 digits)
        df = pd.read_csv(labels_path)
        df['label'] = df['label'].astype(str)
        self.labels_df = df[df['label'].str.len() == 12].reset_index(drop=True)
        print(f"Original size: {len(df)}, Filtered size: {len(self.labels_df)}")

        # constant resizing parameters
        self.IMAGE_HEIGHT = 32
        self.IMAGE_WIDTH = 256

        # transformation for data augmentation & image normalization 
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([self.IMAGE_HEIGHT, self.IMAGE_WIDTH]),
            transforms.RandomAffine(
                degrees=5,        
                translate=(0.05, 0.05)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

        # character mapping to digit
        char_set = "0123456789"
        self.char_map = {char:i for i, char in enumerate(char_set)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        image_i = Image.open(os.path.join(self.image_folder_root, self.labels_df['image_path'][idx]))
        image_i = self.transform(image_i)

        #image_i = image_i.ravel()

        label_i = [self.char_map[char] for char in str(self.labels_df['label'][idx])]
        label_i = torch.tensor(label_i, dtype=torch.long)
        return image_i, label_i