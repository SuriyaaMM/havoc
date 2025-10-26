# Havoc OCR Showdown

This project is an experiment to find the most effective model architecture for a 12-digit number recognition (OCR) task. We explore three different approaches, evolving from a simple linear model to a more complex Convolutional Recurrent Neural Network (CRNN), demonstrating why specialized architectures are critical for sequence-based problems.
**[Kaggle Link](https://www.kaggle.com/competitions/ocr-showdown-decode-12-digit-roll-numbers)**

## Technical Approach

The experiment was conducted by building and training three distinct models on the same dataset.

### 1 `MCLv1` (Baseline: Simple Linear Model)

This model serves as our baseline. It's a simple Multi-Layer Perceptron (MLP) that takes the flattened image as input.

  * **Architecture:** `havoc/registry/modelRegistry/MCLv1`
  * **Result:** This model **failed to learn**. By flattening the image, it destroys all spatial and sequential information. It cannot tell the 1st digit from the 12th. As seen in the logs, its accuracy plateaus at **\~10%**.

### 2 `MCCv1` (Convolutional Model)

This model introduces Convolutional Neural Networks (CNNs) to extract meaningful spatial features before classification.

  * **Architecture:** `havoc/registry/modelRegistry/MCCv1`
  * **Result:** This model **struggles** but eventually learns. After being stuck at \~10% for over 20-30 epochs, it starts to "memorize" the scrambled feature patterns, peaking at **\~78.08% accuracy**.

### 3 `MCCLv1` (CNN + LSTM)

This specialized architecture, commonly known as a CRNN (Convolutional Recurrent Neural Network).

  * **Architecture:** `havoc/registry/modelRegistry/MCCLv1`
  * **Result:** This model is **highly effective**. It starts learning almost immediately and achieves a peak validation accuracy of **99.40%**.

-----

## Results Summary

| Model | Architecture | Peak Validation Accuracy |
| :--- | :--- | :--- |
| `MCLv1` | Simple MLP | \~10.64% |
| `MCCv1` | CNN + MLP | 78.08% |
| `MCCLv1` | **CRNN (CNN + LSTM)** | **99.40%** |

-----

## Installation & Reproduction

This project uses `uv` for fast Python package management.

### Prerequisites

- You must have `uv` installed. 


### Instructions

Follow these steps to create the virtual environment and reproduce the results.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SuriyaaMM/havoc
    cd havoc
    ```

2.  **Create and sync the virtual environment:**

    ```bash
    uv venv
    uv sync
    ```

3.  **Activate the environment and run an experiment:**

    ```bash
    source .venv/bin/activate
    uv run -m havoc.experiment.Ev3
    ```