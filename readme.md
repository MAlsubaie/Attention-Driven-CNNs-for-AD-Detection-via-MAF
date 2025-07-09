# A Novel Deep Learning Approach for Alzheimer’s Disease Detection: Attention-Driven CNNs with Multi-Activation Fusion

## Abstract
Alzheimer's disease (AD) affects over 50 million people worldwide, making early and accurate diagnosis critical for effective treatment and care planning. The diagnosis of AD through neuroimaging classification faces significant challenges. These include subjective clinical evaluations, manual feature extraction requirements, and limited generalisation of automated systems across diverse populations. Recent advances in deep learning, particularly convolutional neural networks (CNNs) and vision transformers (ViTs), have improved diagnostic accuracy. However, these approaches remain constrained by their dependence on large annotated datasets and substantial computational resources. This study proposes a novel attention-enhanced CNNs incorporating a multi-activation fusion (MAF) module, evaluated on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The attention mechanism enables the model to focus on diagnostically significant regions within 3D MRI volumes. The MAF module, inspired by the multi-headed attention concept, employs parallel fully connected layers with distinct activation functions to capture diverse and complementary feature patterns. This design enhances feature expressiveness and improves classification robustness among heterogeneous patient groups. The proposed model achieved 92.1% accuracy with an AUC of 0.99, alongside precision, recall, and F1-scores of 91.3%, 89.3%, and 92%, respectively. Ten-fold cross-validation confirmed model reliability, with consistent performance: 91.23% accuracy, 0.93 AUC, 90.29% precision, and 88.30% recall. Comparative analysis reveals superior performance against several state-of-the-art deep learning models for AD classification. These findings highlight the potential of combining advanced attention mechanisms with multi-activation strategies to improve automated AD diagnosis and enhance diagnostic reliability

## Model Architecture
![Model Architecture](images/arch_diag.png)

## Methodology
![Mmethodology](images/Methodology%20Diagram_2%20-%20Copy.png)


## Important Preprocessing

### HD-BET
This approach is utilized for skull stripping from the images. Preprocess data using the [HD-BET](https://github.com/MIC-DKFZ/HD-BET) method and prepare the data as instructed below.

## Datset Prepration:
Your dataset should be structure in a DataFrame.
```
| ADNI_path | Group |
|-----------|-------|
| ./train/CN/20070215202408595_S13839_I40312.nii.gz | CN |
| ./train/20070215200838637_S22267_I40303.nii.gz | CN |
| ./train/20081008133944436_S13839_I119726.nii.gz | CN |
| ./train/20080410151355187_S47536_I102146.nii.gz | AD |
| ./train20080220165919265_S18766_I91253.nii.gz | AD |
```
## Requirements
- Python 3.9.19
- ```sh
conda create MAF python=3.9.19
```

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/MAlsubaie/Attention-Driven-CNNs-for-AD-Detection-via-MAF.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Attention-Driven-CNNs-for-AD-Detection-via-MAF
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model, run the following command:
```sh
python train.py
```

### Evaluation
To evaluate the model, run the following command:
```sh
python evaluate.py
```

### GradCAM Results
<p align="center">
    <img src="images/GradCAM1.png" alt="GradCAM Result 1" width="45%">
    <img src="images/GradCAM2.png" alt="GradCAM Result 2" width="45%">
</p>


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
