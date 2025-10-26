from dataclasses import dataclass

import torch
from torchvision import transforms

@dataclass
class augmentation:
    image_height: int = 32
    image_width: int = 256

    def __post_init__(self):
        # basic data augmentation 
        self.transformA = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([self.image_height, self.image_width]),
            transforms.ToTensor()
        ])
        # a bit more sophisticated data augmentation
        self.transformB = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([self.image_height, self.image_width]),
            transforms.RandomAffine(
                degrees=5,        
                translate=(0.05, 0.05)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
             transforms.ToTensor()
        ])


def splitDataset(
    logger,
    image_folder_root: str,
    labels_path: str,
    split_length: list,
    dataset: torch.utils.data.Dataset,
    train_transform,
    test_transform,
):
    fullDataset = dataset(
        logger,
        image_folder_root,
        labels_path,
        None
    )

    trainSize = int(split_length[0] * len(fullDataset))
    testSize = int(split_length[1] * len(fullDataset))

    trainIndices, testIndices = torch.utils.data.random_split(
        range(len(fullDataset)),
        [trainSize, testSize]   
    )

    trainDataset = dataset(
        logger,
        image_folder_root,
        labels_path,
        train_transform
    ) 

    testDataset = dataset(
        logger,
        image_folder_root,
        labels_path,
        test_transform
    )

    trainDataset = torch.utils.data.Subset(trainDataset, trainIndices.indices)
    testDataset = torch.utils.data.Subset(testDataset, testIndices)

    return trainDataset, testDataset

