# Spatio-Temporal Wildfire Prediction using an FT-Transformer

## Project Objective
This project develops a high-performance deep learning model to predict the confidence level of wildfire detections from satellite imagery data. By engineering innovative spatio-temporal features and leveraging a state-of-the-art FT-Transformer architecture, the model achieves high accuracy and F1-scores, demonstrating its potential for real-world environmental monitoring and disaster response systems.

## Key Features
- **Spatio-Temporal Feature Engineering**: Enriches the dataset by creating features that describe the spatial neighborhood of each fire detection, providing crucial context to the model.
- **Advanced Deep Learning**: Implements an `FT-Transformer`, a modern Transformer-based architecture specifically designed for tabular data, built with PyTorch.
- **Imbalanced Data Handling**: Employs a sophisticated hybrid sampling strategy (SMOTE for over-sampling minorities, RandomUnderSampler for under-sampling the majority) to train a robust and unbiased model.
- **End-to-End Pipeline**: Provides a complete, reproducible pipeline from raw data combination and cleaning to model training, evaluation, and interpretation.

## Dataset Description
The dataset is an aggregation of VIIRS satellite fire detection data from 2012 to 2023 for Saudi Arabia, sourced from Kaggle. Each row represents a single fire detection event at a specific point in time and space.

- **Spatial Features**: `latitude`, `longitude`
- **Temporal Features**: `acq_date` (date of acquisition), `acq_time` (time of acquisition)
- **Sensor Features**: `bright_ti4`, `bright_ti5` (brightness temperature in Kelvin), `frp` (Fire Radiative Power)
- **Target Variable**: `confidence` (a categorical variable indicating the confidence of the fire detection: Nominal, High, or Low). This was mapped to numerical values `0`, `1`, `2` respectively.

## Methodology

The project pipeline is executed in a sequential and logical manner:

### 1. Data Preprocessing & Feature Engineering
- **Data Consolidation**: Multiple yearly CSV files were combined into a single master dataset.
- **Cleaning**: Irrelevant columns were dropped, and rows with missing values were removed.
- **Temporal Features**: The `acq_date` and `acq_time` columns were used to engineer more useful features: `hour`, `month`, and `year`.
- **Encoding & Scaling**: Numerical features (like brightness and FRP) were standardized using `StandardScaler`. Categorical features (like `daynight` and the new temporal features) were one-hot encoded.

### 2. Spatio-Temporal Feature Engineering
This is the core innovation of the project. A standard model would treat each fire detection as an independent event. However, fires are a spatial phenomenon. A detection is more likely to be significant if it's surrounded by other high-intensity detections.

To capture this, I implemented a function that:
1.  Uses a `sklearn.neighbors.BallTree` with a `haversine` metric to efficiently find the **k-nearest spatial neighbors** for every single data point.
2.  For each point, it calculates the **mean and standard deviation** of the *features* of its neighbors.
3.  These new aggregated features (e.g., `mean_frp_k32`, `std_bright_ti4_k32`) are appended to the original feature set, providing the model with rich contextual information about the spatial environment of each detection.

### 3. Handling Extreme Class Imbalance
The dataset is highly imbalanced, with the 'Nominal' confidence class vastly outnumbering the 'High' and 'Low' classes. To prevent the model from becoming biased, a hybrid sampling strategy was applied to the **training data only**:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to generate synthetic samples for the minority 'High' and 'Low' confidence classes.
- **RandomUnderSampler**: Used to reduce the number of samples in the majority 'Nominal' confidence class.
This creates a more balanced dataset for the model to learn from effectively.

### 4. Model Architecture: FT-Transformer
Instead of a traditional model like XGBoost or a simple MLP, this project uses an **FT-Transformer** (`rtdl.FTTransformer`), a state-of-the-art deep learning model for tabular data.
- **Feature Tokenizer**: The model begins by converting all numerical and categorical features into high-dimensional embeddings (tokens), similar to how words are processed in NLP.
- **Transformer Blocks**: These embeddings are then passed through a series of Transformer blocks. Each block uses a **self-attention mechanism**, allowing every feature to weigh its importance against every other feature. This enables the model to learn incredibly complex and non-linear relationships within the data.
- **The architecture used**: 4 Transformer blocks, with a token dimension of 192.

### 5. Training & Evaluation
- **Framework**: The model was built and trained using **PyTorch**.
- **Loss Function**: `CrossEntropyLoss` was used with **class weights** to give more importance to the minority classes during training, further mitigating the effects of class imbalance.
- **Optimizer & Scheduler**: `AdamW` optimizer was paired with a `CosineAnnealingLR` scheduler, which helps the model converge to a better minimum by gradually decreasing the learning rate over epochs.
- **Evaluation**: The model was evaluated on an unseen test set using **Accuracy** and, more importantly, **Macro F1-Score**, which provides a better measure of success on imbalanced datasets.

## Results & Conclusion

The model achieved excellent performance on the unseen test set:
- **Accuracy**: 0.9839
- **Macro F1-Score**: 0.9376

**Classification Report:**```
              precision    recall  f1-score   support

           0       1.00      0.99      0.99     82217
           1       0.91      0.97      0.94      1611
           2       0.81      0.97      0.88      5405

    accuracy                           0.98     89233
   macro avg       0.91      0.97      0.94     89233
weighted avg       0.99      0.98      0.98     89233
```
The high recall across all classes, especially the minority classes (97% for both 'High' and 'Low' confidence), indicates that the model is exceptionally effective at identifying all types of fire events correctly.

The combination of advanced spatio-temporal feature engineering and a powerful Transformer architecture proved to be a highly successful strategy for this complex prediction task.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Wildfire_Confidence_Prediction_FT-Transformer
    ```
2.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place data:** Create a `data/` folder in the root directory and place the raw `.csv` dataset files inside it.

4.  **Execute the script:**
    The script will perform all steps from data combination to final evaluation and print the results to the console.
