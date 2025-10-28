import torch
import numpy as np
import pandas as pd
import matplotlib
import PIL
import os
from dataclasses import dataclass, field

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
    labelsPath: str = Environment.env["LABELS_PATH"]
    imageFolderPath: str = Environment.env["IMAGE_FOLDER_ROOT"]

    batchSize: int = 64
    learningRate: float = 3e-4
    numWorkers: int = 8
    epochs: int = 50

    split: list = field(default_factory=lambda: [0.8, 0.2]) 
    
    aftermathImageHeight: int = 32
    aftermathImageWidth: int = 256

    experimentName: str = "Ev1"
    modelRegistryPath: str = Environment.env["MODEL_REGISTRY"]
    onnxRegistryPath: str =  Environment.env["ONNX_REGISTRY"]
    dfRegistryPath: str = Environment.env["DF_REGISTRY"]
    onnxVersion: int = 24

    @property
    def modelFileName(self) -> str:
        return os.path.join(self.modelRegistryPath, f"{self.experimentName}.pth")
        
    @property
    def onnxFileName(self) -> str:
        return os.path.join(self.onnxRegistryPath, f"{self.experimentName}.onnx")
    
    @property
    def dfFileName(self) -> str:
        return os.path.join(self.dfRegistryPath, f"{self.experimentName}.csv")