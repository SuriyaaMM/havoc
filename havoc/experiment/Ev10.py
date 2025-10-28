import torch
import torch.nn as nn
import pandas as pd

from havoc.registry.modelRegistry import MCCv3
from havoc.registry.datasetRegistry import OCRIDv2
from havoc.utils.logger import LogSubsystem
from havoc.utils.cfg import dumpReprodInfo, staticConfig
from havoc.utils.data import augmentation, splitDataset

# initialize logging subsystem
LogSubsystem.setConsoleLogger()
LogSubsystem.setFileLogger("Ev10.txt")
logger = LogSubsystem.getBaseLogger()

# dump version info (reproduceability)
dumpReprodInfo(logger)

# initialize configuration
config = staticConfig()
config.experimentName = "Ev10"
config.epochs = 200

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
    OCRIDv2,
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

model = MCCv3()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learningRate, weight_decay=3e-2)
loss_function = nn.CrossEntropyLoss()

minValidationLoss = float('inf')
maxValidationAccuracy = 0.0

cnnGradHistory = []
mlpGradHistory = []
trainingLossHistory = []
validationLossHistory = []
validationAccHistory = []

for epoch in range(config.epochs):
    trainingLoss = 0.0
    cnnNorm = 0.0
    mlpNorm = 0.0
    cnnNorms = []
    mlpNorms = []
    logger.info(f"TOTAL EPOCHS: {config.epochs}")
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

        # gradient norm calculation
        cnnParams = [p for p in model.fcnn.parameters() if p.grad is not None]
        mlpParams = [p for p in model.fmlp.parameters() if p.grad is not None]

        if cnnParams:
            cnnNorms.append(torch.sqrt(sum(torch.sum(p.grad ** 2) for p in cnnParams)).item())
        if mlpParams:
            mlpNorms.append(torch.sqrt(sum(torch.sum(p.grad ** 2) for p in mlpParams)).item())

        optimizer.step()

    logger.info(f"training loss: {trainingLoss / len(trainDataloader):.4f}")
    trainingLossHistory.append(trainingLoss / len(trainDataloader))
    cnnGradHistory.append(sum(cnnNorms) / len(cnnNorms))
    mlpGradHistory.append(sum(mlpNorms) / len(mlpNorms))

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
    validationLossHistory.append(validationLoss / len(testDataloader))
    validationAccHistory.append(validationAccuracy)

    # save the model if it meets the criteria
    if(validationAccuracy > maxValidationAccuracy) or (minValidationLoss > validationLoss):
        maxValidationAccuracy = validationAccuracy
        minValidationLoss = validationLoss

        torch.save(model.state_dict(), config.modelFileName)

        logger.info(f"best accuracy: {maxValidationAccuracy:.4f}% | best loss: {minValidationLoss:.4f}")
        logger.info("saved to model registry!")

df = pd.DataFrame({
    "cnnGradHistory" : cnnGradHistory,
    "mlpGradHistory" : mlpGradHistory,
    "trainingLossHistory" : trainingLossHistory,
    "validationLossHistory" : validationLossHistory,
    "validationAccHistory" : validationAccHistory
})

df.to_csv(config.dfFileName)
