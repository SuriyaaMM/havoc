import os
import logging

import torch
import pandas as pd
from PIL import Image

# OCR Image Dataset Version 1
class OCRIDv1(torch.utils.data.Dataset):
    def __init__(
        self, 
        logger: logging.Logger,
        image_folder_root: str, 
        labels_path: str,
        transform
    ):
        super().__init__()
        self.imageFolderRoot = image_folder_root
        self.transform = transform

         # labels dataframe (remove digits which are not of 12 digits)
        df = pd.read_csv(labels_path)
        df['label'] = df['label'].astype(str)
        self.labelsDF = df[df['label'].str.len() == 12].reset_index(drop=True)
        logger.info(f"Original size: {len(df)}, Filtered size: {len(self.labelsDF)}")

        # character mapping to digit
        charSet = "0123456789"
        self.charMap = {char:i for i, char in enumerate(charSet)}

    def __len__(self):
        return len(self.labelsDF)

    def __getitem__(self, idx):
        imageI = Image.open(os.path.join(self.imageFolderRoot, self.labelsDF['image_path'][idx]))
        imageI = self.transform(imageI)

        imageI = imageI.ravel()

        labelI = [self.charMap[char] for char in str(self.labelsDF['label'][idx])]
        labelI = torch.tensor(labelI, dtype=torch.long)
        return imageI, labelI

# OCR Image Dataset Version 2
class OCRIDv2(torch.utils.data.Dataset):
    def __init__(
        self, 
        logger: logging.Logger,
        image_folder_root: str, 
        labels_path: str,
        transform
    ):
        super().__init__()
        self.imageFolderRoot = image_folder_root
        self.transform = transform

        # labels dataframe (remove digits which are not of 12 digits)
        df = pd.read_csv(labels_path)
        df['label'] = df['label'].astype(str)
        self.labelsDF = df[df['label'].str.len() == 12].reset_index(drop=True)
        logger.info(f"Original size: {len(df)}, Filtered size: {len(self.labelsDF)}")
        
        # character mapping to digit    
        charSet = "0123456789"
        self.charMap = {char:i for i, char in enumerate(charSet)}

    def __len__(self):
        return len(self.labelsDF)

    def __getitem__(self, idx):
        imageI = Image.open(os.path.join(self.imageFolderRoot, self.labelsDF['image_path'][idx]))
        imageI = self.transform(imageI)

        labelI = [self.charMap[char] for char in str(self.labelsDF['label'][idx])]
        labelI = torch.tensor(labelI, dtype=torch.long)
        return imageI, labelI
    
# OCR Image Dataset Version 3
class OCRIDv3(torch.utils.data.Dataset):
    def __init__(
        self, 
        logger: logging.Logger,
        image_folder_root: str, 
        labels_path: str,
        transform
    ):
        super().__init__()
        self.imageFolderRoot = image_folder_root
        self.labelsDF = pd.read_csv(labels_path)
        self.transform = transform

        # character mapping to digit    
        charSet = "0123456789"
        self.charMap = {char:i for i, char in enumerate(charSet)}
        logger.info(f"Initialized dataset with {len(self.labelsDF)} elements")

    def __len__(self):
        return len(self.labelsDF)

    def __getitem__(self, idx):
        imageI = Image.open(os.path.join(self.imageFolderRoot, self.labelsDF['image_path'][idx]))
        imageI = self.transform(imageI)

        labelI = [self.charMap[char] for char in str(self.labelsDF['label'][idx])]
        labelI = torch.tensor(labelI, dtype=torch.long)

        label_length = torch.tensor(len(labelI), dtype=torch.long)
        return imageI, labelI, label_length