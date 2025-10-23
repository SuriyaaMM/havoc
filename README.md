# OCR Showdown (12 Digit Digit Recognition)

### Project Objective

The objective of this project is to design, implement, and iteratively improve a deep learning model for Optical Character Recognition (OCR). The specific task is to correctly identify a 12-digit numerical string from a given dataset of images.

This document details the step-by-step progression, starting from a naive baseline model and culminating in a robust, regularized Convolutional Neural Network (CNN) that successfully learns from augmented data.

-----

### Installation and Setup

- Pre-requisite : [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Dataset: [Kaggle](https://www.kaggle.com/competitions/ocr-showdown-decode-12-digit-roll-numbers/data)

```bash
git clone https://github.com/SuriyaaMM/OCR_Showdown
cd OCR_Showdown

# Synchronize the Packages
uv sync

# Activate the Virtual Environment
source .venv/bin/activate
```
-----

### Code Structure

The project is organized into several key files:

  * `ExperimentV5.py`: The main script for training and evaluating the final `ClassifierV5` model. It contains the complete training loop, validation loop, and model saving logic.
  * `ModelRegistry.py`: Contains all PyTorch model definitions:
      * `ClassifierV1`: Naive MLP (Baseline).
      * `ClassifierV2`: Baseline CNN.
      * `ClassifierV4`: Deep CNN (unstable).
      * `ClassifierV5`: Final robust CNN with `BatchNorm`.
  * `DatasetRegistry.py`: Contains all `torch.utils.data.Dataset` definitions:
      * `OCRImageDatasetV1`: For the MLP (flattens images).
      * `OCRImageDatasetV2`: For the baseline CNN (resizes images).
      * `OCRImageDatasetV4`: Adds `RandomAffine` and `ColorJitter` augmentations.
  * `requirements.txt`: Lists all Python dependencies for `uv sync`.
  * `model_registry/`: Output directory where the best `.pth` (PyTorch model weights) are saved.
  * `onnx_registry/`: Output directory where the best models are exported to the `.onnx` format for inference.
-----

### The Iterative Progression

This project's development followed four distinct stages, demonstrating a clear path of iterative problem-solving.

#### Stage 1: The Naive MLP (V1)

  * **Model:** `ClassifierV1`
  * **Dataset:** `OCRImageDatasetV1`
  * **Method:** The first baseline was a simple Multi-Layer Perceptron (MLP). The dataset flattens each image (`.ravel()`) into a 1D vector and feeds it to `nn.Linear` layers.
  * **Result:** **Failure**. Accuracy was stuck at \~10% (random guessing). Flattening the image destroys all 2D spatial information, making it impossible for the model to learn the "shapes" of the digits.

#### Stage 2: The CNN Baseline (V2)

  * **Model:** `ClassifierV2`
  * **Dataset:** `OCRImageDatasetV2`
  * **Method:** Realizing that spatial data is key, the architecture was switched to a **Convolutional Neural Network (CNN)**. The dataset was modified to resize all images to a uniform `(32, 256)` and feed them as 2D tensors to the model.
  * **Result:** **Success**. This model quickly broke the 10% barrier and achieved high accuracy (\~87%). This confirmed the CNN was the correct architecture.

#### Stage 3: The Unstable Deep Model (V4)

  * **Model:** `ClassifierV4`
  * **Dataset:** `OCRImageDatasetV4`
  * **Method:** To improve on the V2 baseline, two changes were made:
    1.  **Data Augmentation:** `RandomAffine` and `ColorJitter` were added to the dataset to create more varied training data and prevent overfitting.
    2.  **Deeper Network:** The fully-connected head of the classifier was made much deeper.
  * **Result:** **Failure**. The model's accuracy collapsed back to \~10%. The combination of a very deep network and "harder" augmented data led to **extreme training instability**. The gradients likely vanished or exploded, preventing the model from learning.

#### Stage 4: The Robust & Regularized Model (V5)

  * **Model:** `ClassifierV5`
  * **Dataset:** `OCRImageDatasetV4` (The augmented one)
  * **Method:** This model was designed to fix the instability of V4.
    1.  **`nn.BatchNorm`**: `BatchNorm2d` and `BatchNorm1d` were added after *every* `Conv` and `Linear` layer. This normalized the activations, stabilized the gradients, and allowed the deep network to converge.
    2.  **`Dropout` Order**: `Dropout` layers were confirmed to be placed *after* `ReLU` activations to provide regularization without killing the training signal.
  * **Result:** **Success**. This model successfully trains on the augmented data, achieving the highest accuracy (\~90%+) and demonstrating the best generalization.

-----

### Performance Summary

The training results clearly show the impact of each architectural decision.

| Model Version | Dataset Used | Architecture | Key Change | Validation Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| `ClassifierV1` | `OCRImageDatasetV1` | MLP (Linear) | Naive attempt on flattened images. | \10.13% (Failed) |
| `ClassifierV2` | `OCRImageDatasetV2` | CNN | **Switched to CNN**; standardized image size. | **\85.60** |
| `ClassifierV4` | `OCRImageDatasetV4` | Deep CNN + Aug | Added augmentation; **No `BatchNorm`**. | \10.00% (Failed) |
| `ClassifierV5` | `OCRImageDatasetV4` | CNN + BN + Dropout | **Added `BatchNorm`** & fixed `Dropout` order. | **\92.3%+ (Best)** |

-----

### Conclusion

This project successfully demonstrates the iterative process of model development. The key takeaway is that while a correct baseline architecture (CNN) is essential, **stabilization techniques like `BatchNorm` are critical** for enabling deep networks to learn from complex, augmented data. The final `ClassifierV5` model represents a stable, robust, and well-regularized solution to the OCR task.