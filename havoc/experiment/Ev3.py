import torch
import torch.nn as nn
import torch.nn.functional as F

from havoc.registry.modelRegistry import MCCLv1
from havoc.registry.datasetRegistry import OCRIDv3
from havoc.utils.logger import LogSubsystem
from havoc.utils.cfg import dumpReprodInfo, staticConfig
from havoc.utils.data import augmentation, splitDataset

# initialize logging subsystem
LogSubsystem.setConsoleLogger()
LogSubsystem.setFileLogger("Ev3.txt")
logger = LogSubsystem.getBaseLogger()

# dump version info (reproduceability)
dumpReprodInfo(logger)

# initialize configuration
config = staticConfig()
config.experimentName = "Ev3"

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
    OCRIDv3,
    aug.transformB,
    aug.transformA
)

def ctcCollate(batch: torch.Tensor):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labelLength = [item[2] for item in batch]

    imagesBatch = torch.stack(images, dim=0)
    labelsBatch = torch.cat(labels, dim=0)
    labelsLengthBatch = torch.stack(labelLength)

    return imagesBatch, labelsBatch, labelsLengthBatch

# initialize loaders
trainDataloader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=config.batchSize,
                                              num_workers=config.numWorkers,
                                              persistent_workers=True,
                                              collate_fn=ctcCollate)
testDataloader = torch.utils.data.DataLoader(testDataset,
                                              batch_size=config.batchSize,
                                              num_workers=config.numWorkers,
                                              persistent_workers=True,
                                              collate_fn=ctcCollate)

device = torch.device("cuda")
model = MCCLv1()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learningRate)
lossFunction = nn.CTCLoss(blank=10, zero_infinity=True)

minValidationLoss = float('inf')
maxValidationAccuracy = 0.0

for epoch in range(config.epochs):
    trainingLoss = 0.0
    logger.info("-"*50)
    logger.info(f"epoch {epoch + 1} started")

     # ----- training part
    images: torch.Tensor
    labels: torch.Tensor
    labelLengths: torch.Tensor
   
    model.train()
    for images, labels, labelLengths in trainDataloader:
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)
        labelLengths = labelLengths.to(device)

        y: torch.Tensor = labels
        yHat: torch.Tensor = model(images)

        batchSize = images.shape[0]
        seqLength = yHat.shape[1]

        # [B, T, C]
        # ctc expects [T, B, C]
        yHat = yHat.permute([1, 0, 2])

        inputLengths = torch.full(size=(batchSize, ), fill_value=seqLength, dtype=torch.long).to(device)
        targetLengths = labelLengths
        logProbs = F.log_softmax(yHat, dim=2)
        loss: torch.Tensor = lossFunction(logProbs, y, inputLengths, targetLengths)

        loss.backward()
        optimizer.step()

        trainingLoss += loss.item()
    
    logger.info(f"training loss: {trainingLoss / len(trainDataloader):.4f}")

    # ----- validation part
    model.eval()

    totalCorrect = 0
    totalSamples = 0
    validationLoss = 0.0
    validationAccuracy = 0.0
    
    with torch.no_grad():
        for images, labels, labelLengths in testDataloader:
            images = images.to(device)
            labels = labels.to(device)
            labelLengths = labelLengths.to(device)

            y: torch.Tensor = labels
            yHat: torch.Tensor = model(images)

            batchSize = images.shape[0]
            seqLength = yHat.shape[1]

            yHat = yHat.permute([1, 0, 2])

            inputLengths = torch.full(size=(batchSize, ), fill_value=seqLength, dtype=torch.long).to(device)
            targetLengths = labelLengths
            logProbs = F.log_softmax(yHat, dim=2)
            loss = lossFunction(logProbs, y, inputLengths, targetLengths)

            validationLoss += loss.item()

            # ----- accuracy calculation (decoder)
            yHat = yHat.permute([1, 0, 2])
            predIndices = torch.argmax(yHat, dim=2)

            gTruthLabels = []
            start = 0
            for length in targetLengths:
                gTruthLabels.append(labels[start:start+length].tolist())
                start += length

            for i in range(batchSize):
                rawPredLogits = predIndices[i].tolist()
                gTruthLabel = gTruthLabels[i]

                # ----- greedy decoding
                decodedPred = []
                for l in range(len(rawPredLogits)):
                    # skip consecutive characters
                    if l > 0 and rawPredLogits[l] == rawPredLogits[l-1]:
                        continue
                    # skip blank token
                    if rawPredLogits[l] == 10:
                        continue

                    decodedPred.append(rawPredLogits[l])

                if decodedPred == gTruthLabel:
                    totalCorrect += 1 

                totalSamples += 1

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
