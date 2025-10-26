import torch
import torch.nn as nn

from havoc.registry.modelRegistry import MCLv1
from havoc.registry.datasetRegistry import OCRIDv1
from havoc.utils.logger import LogSubsystem
from havoc.utils.cfg import dumpReprodInfo, staticConfig
from havoc.utils.data import augmentation, splitDataset

# initialize logging subsystem
LogSubsystem.setConsoleLogger()
LogSubsystem.setFileLogger("Ev1.txt")
logger = LogSubsystem.getBaseLogger()

# dump version info (reproduceability)
dumpReprodInfo(logger)

# initialize configuration
config = staticConfig()
config.experimentName = "Ev1"
config.epochs = 100

# initialize data augmentation
aug = augmentation(
    image_height=config.aftermathImageHeight,
    image_width=config.aftermathImageWidth
)

# split the dataset
trainDataset, testDataset = splitDataset(
    logger,
    config.imageFolderPath,
    config.labelsPath,
    config.split,
    OCRIDv1,
    aug.transformB,
    aug.transformA
)

# initialize loaders
trainDataloader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=config.batchSize,
                                              num_workers=config.numWorkers,
                                              persistent_workers=True)
testDataloader = torch.utils.data.DataLoader(testDataset,
                                              batch_size=config.batchSize,
                                              num_workers=config.numWorkers,
                                              persistent_workers=True)


device = torch.device("cuda")

model = MCLv1()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learningRate)
loss_function = nn.CrossEntropyLoss()

minValidationLoss = float('inf')
maxValidationAccuracy = 0.0

for epoch in range(config.epochs):
    trainingLoss = 0.0
    logger.info("-"*50)
    logger.info(f"epoch {epoch + 1} started")
    
    # ----- training part
    images: torch.Tensor
    labels: torch.Tensor

    model.train()
    for images, labels in trainDataloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        yHat = model(images)
        y = labels

        # reshape to 10 output classes 
        yHat = yHat.view(-1, 10)
        y = y.view(-1)

        loss: torch.Tensor = loss_function(yHat, y)
        trainingLoss += loss.item()
        loss.backward()
        optimizer.step()

    logger.info(f"training loss: {trainingLoss / len(trainDataloader):.4f}")

    # ----- validation part
    validationLoss = 0.0
    totalCorrect = 0
    totalSamples = 0
    
    model.eval()  
    with torch.no_grad():  
        for images, labels in testDataloader:
            images = images.to(device)
            labels = labels.to(device)
        
            yHat = model(images)
            y = labels
           
            yHat = yHat.view(-1, 10)
            y = y.view(-1)
           
            loss = loss_function(yHat, y)
            validationLoss+= loss.item()
            
            _, predicted = torch.max(yHat, 1)
            totalSamples += y.size(0)
            totalCorrect += (predicted == y).sum().item()

    validationAccuracy = (totalCorrect/totalSamples) * 100.0
    logger.info(f"validation loss: {validationLoss / len(testDataloader):.4f}")
    logger.info(f"validation accuracy: {validationAccuracy:.4f}%")

    # save the model if it meets the criteria
    if(validationAccuracy > maxValidationAccuracy) or (minValidationLoss > validationLoss):
        maxValidationAccuracy = validationAccuracy
        minValidationLoss = validationLoss

        torch.save(model.state_dict(), config.modelFileName)

        logger.info(f"best accuracy: {maxValidationAccuracy:.4f}% | best loss: {minValidationLoss:.4f}")
        logger.info("saved to model registry!")