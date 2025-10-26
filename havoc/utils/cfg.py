import torch
import numpy as np
import pandas as pd
import matplotlib
import PIL
import os
from dataclasses import dataclass

from havoc.utils.env import Environment

def dumpReprodInfo(logger):
    logger.info("-"*50)
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"numpy version: {np.__version__}")
    logger.info(f"pandas version: {pd.__version__}")
    logger.info(f"matplotlib version: {matplotlib.__version__}")
    logger.info(f"pillow version: {PIL.__version__}")
    logger.info("-"*50)
    logger.info(f"CUDA Is Available : {torch.cuda.is_available()}")
    logger.info("-"*50)

@dataclass()
class staticConfig(object):
    labelsPath = Environment.env["LABELS_PATH"]
    imageFolderPath = Environment.env["IMAGE_FOLDER_ROOT"]

    batchSize = 64
    learningRate = 3e-4
    numWorkers = 8
    epochs = 50
    split = [0.8, 0.2]
    
    aftermathImageHeight = 32
    aftermathImageWidth = 256

    experimentName = "Ev1"
    modelRegistryPath = Environment.env["MODEL_REGISTRY"]
    modelFileName = os.path.join(modelRegistryPath, f"{experimentName}.pth")
    onnxRegistryPath =  Environment.env["ONNX_REGISTRY"]
    onnxFileName = os.path.join(onnxRegistryPath, f"{experimentName}.onnx")
    onnxVersion = 24