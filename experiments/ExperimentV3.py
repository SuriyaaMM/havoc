import numpy as np 
import pandas as pd
import torch
import torch.export
import torch.nn as nn
import PIL
import matplotlib
import os

from ModelRegistry import ClassifierV3
from DatasetRegistry import OCRImageDatasetV2

print(f"pytorch version: {torch.__version__}")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"pillow version: {PIL.__version__}")
print("-"*50)
print(f"CUDA Is Available : {torch.cuda.is_available()}")
print("-"*50)

labels_path = "roll_number_dataset2/labels.csv"
image_folder_root = "roll_number_dataset2/"

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_WORKERS = 4
EPOCHS = 50
SPLIT = [0.8, 0.2]
EXPERIMENT_NAME = "ExperimentV3"
MODEL_REGISTRY_PATH = os.path.join(os.getcwd(), "model_registry")
MODEL_NAME = os.path.join(MODEL_REGISTRY_PATH, f"{EXPERIMENT_NAME}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}.pth")
ONNX_REGISTRY_PATH = os.path.join(os.getcwd(), "onnx_registry")
ONNX_NAME = os.path.join(ONNX_REGISTRY_PATH, f"{EXPERIMENT_NAME}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}.onnx")
ONNX_VERSION = 24

dataset = OCRImageDatasetV2(image_folder_root, labels_path)
train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            lengths=SPLIT)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=NUM_WORKERS,
                                              persistent_workers=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=NUM_WORKERS,
                                              persistent_workers=True)


device = torch.device("cuda")

model = ClassifierV3()
print(model)
print("-"*50)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

debug = False

print("starting training!")
print("-"*50)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Split : {SPLIT}")
print("-"*50)
print(f"making model registry at {MODEL_REGISTRY_PATH}")
os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)
print(f"making onnx registry at {ONNX_REGISTRY_PATH}")
os.makedirs(ONNX_REGISTRY_PATH, exist_ok=True)
print("-"*50)

prev_validation_loss = float('inf')
prev_validation_accuracy = 0.0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    print(f"epoch {epoch + 1} started: ", end = " ")

    for images, labels in train_dataloader:
        # zero grad
        optimizer.zero_grad()
        # send to device
        images = images.to(device)
        labels = labels.to(device)

        if debug:
            print(f"images.shape : {images.shape}")
            print(f"labels.shape : {labels.shape}")

        # forward propagate
        y_hat = model(images)
        y = labels

        if debug:
            print(f"y.shape : {y.shape}")
            print(f"y_hat.shape : {y_hat.shape}") 

        # tensor reshaping
        y_hat = y_hat.view(-1, 10)
        y = y.view(-1)

        if debug: 
            print(f"y.shape (reshaped): {y.shape}")
            print(f"y_hat.shape (reshaped): {y_hat.shape}")
    
        loss = loss_function(y_hat, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    validation_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    model.eval()  
    with torch.no_grad():  
        for images, labels in test_dataloader:
            optimizer.zero_grad()
            # send to device
            images = images.to(device)
            labels = labels.to(device)
            # forward propagate
            y_hat = model(images)
            y = labels
            # tensor reshaping
            y_hat_reshaped = y_hat.view(-1, 10)
            y_reshaped = y.view(-1)
            # loss calculation
            loss = loss_function(y_hat_reshaped, y_reshaped)
            validation_loss += loss.item()
            # validation calculation
            _, predicted = torch.max(y_hat_reshaped, 1)
            total_samples += y_reshaped.size(0)
            total_correct += (predicted == y_reshaped).sum().item()

    avg_val_loss = validation_loss / len(test_dataloader)
    validation_accuracy = (total_correct / total_samples) * 100.0

    # save the model if validation accuracy & validation loss are better
    if (prev_validation_loss > avg_val_loss) and (prev_validation_accuracy < validation_accuracy):
        # pytorch state dict exporting
        torch.save(model.state_dict(), MODEL_NAME)
        dummy_input = torch.randn(1, 1, 
                                  32,
                                  256,
                                  device=device
        )

        # onnx exporting
        dynamic_shapes = {
            "x": {0: torch.export.Dim("batch_size")}
        }
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_NAME,
            export_params=True,
            opset_version=ONNX_VERSION,
            do_constant_folding=True,
            input_names=['input'],      
            output_names=['output'],     
            dynamic_shapes=dynamic_shapes
        )
        print("model & onnx files are saved at registry | ", end = " ")

    prev_validation_accuracy = validation_accuracy
    prev_validation_loss = avg_val_loss

    model.train()  
    print(f"loss = {epoch_loss:.4f} | val_loss = {avg_val_loss:.4f} | val_accuracy = {validation_accuracy:.2f}%")