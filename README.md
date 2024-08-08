# Text Classification Project with BERT

## Project Overview

This project is aimed at developing a text classification system using the BERT (Bidirectional Encoder Representations from Transformers) model. BERT is a pre-trained transformer model known for its superior performance in natural language processing tasks. The objective of this project is to classify text data into predefined categories accurately.

## Project Workflow

## 1. Importing Libraries

- **datasets**: For handling datasets, importing `Dataset` and `DatasetDict` from the `datasets` library.
- **sklearn.preprocessing**: For label encoding, importing `LabelEncoder`.
- **numpy** and **pandas**: For data manipulation and handling.
- **torch**: PyTorch library for deep learning.
- **transformers**: For working with pre-trained models, including `AutoTokenizer`, `AutoModel`, and `AutoModelForSequenceClassification`.
- **sklearn.metrics**: For evaluating model performance using metrics like `accuracy_score` and `f1_score`.

## 2. Loading Data

- **DataFrames Creation**:
  - `train_df` and `test_df`: DataFrames are created by reading CSV files named `train_data.csv` and `test.csv`. These files contain text data (`comment`) and their associated labels.

## 3. Encoding Labels

- **Label Encoding**:
  - `class_encoding`: A dictionary maps text labels (`'CAG'`, `'NAG'`, `'OAG'`) to numeric values (`0`, `1`, `2`). This encoding is necessary for machine learning models that require numerical input.
  - The `map` function is used to apply this encoding to the `class_label` column in both the training and testing DataFrames, creating a new column `label` with numeric values.

## 4. Preparing Datasets

- **Dataset Preparation**:
  - The DataFrames are filtered to include only the `comment` and `label` columns.
  - `Dataset.from_pandas`: Converts the DataFrames into Hugging Face Dataset objects.
  - `DatasetDict`: Combines the training and testing datasets into a dictionary format, which is easier to manage for training and evaluation.

## 5. Tokenization

- **Model Checkpoint**:
  - `model_ckpt`: Specifies the model checkpoint for `indic-bert`, a pre-trained model.

- **Tokenizer**:
  - `AutoTokenizer.from_pretrained`: Loads the tokenizer associated with the specified model checkpoint. The `keep_accents=True` parameter ensures that accents in the text are preserved.

- **Tokenize Function**:
  - Defines how the text should be tokenized. It pads and truncates the text to a maximum length of 512 tokens to fit the input requirements of the model.
  - `dataset.map`: Applies the `tokenize` function to each example in the dataset, encoding the text into a format suitable for the model.

## 6. Model Setup

- **Imports**:
  - `torch`: PyTorch library for tensor operations and model handling.
  - `AutoModel` from `transformers`: For loading pre-trained models.

- **Model Checkpoint**:
  - `model_ckpt`: Specifies the pre-trained model checkpoint `'ai4bharat/indic-bert'`. This is a model trained on Indic languages.

- **Device Configuration**:
  - `device`: Checks if a GPU (cuda) is available. If not, it defaults to CPU. This allows the code to leverage GPU acceleration if available, improving computation speed.

- **Model Loading**:
  - `AutoModel.from_pretrained`: Loads the pre-trained model specified by `model_ckpt` and moves it to the appropriate device (GPU or CPU).
  - `model.config`: Displays the configuration details of the loaded model, such as hidden size, number of attention heads, etc. This information is useful for understanding the model architecture.

## 7. Extracting Hidden States

- **Function Definition**:
  - `extract_hidden_states`: This function extracts the hidden states (internal representations) of the input text from the pre-trained model.
    - `inputs`: Prepares input tensors by moving them to the configured device.
    - `torch.no_grad()`: Disables gradient calculations to save memory and computations during inference.
    - `model(**inputs)`: Passes the inputs through the model to obtain the hidden states.
    - `last_hidden_state[:,0].cpu().numpy()`: Extracts the hidden state for the first token of each input sequence, moves it to CPU, and converts it to a NumPy array.

- **Applying the Function**:
  - `dataset_encoded.map(extract_hidden_states, batched=True)`: Applies the `extract_hidden_states` function to each example in the dataset, processing batches of data.

## 8. Checking for MPS (Metal Performance Shaders)

- **MPS Device Check**:
  - `torch.backends.mps.is_available()`: Checks if an MPS device (used for acceleration on Apple hardware) is available.
  - `mps_device = torch.device("mps")`: If available, sets the device to MPS.

## 9. Re-loading the Model on MPS

- **Device Configuration**:
  - Reconfigures the device to MPS if available; otherwise, defaults to CPU.

- **Re-loading Model**:
  - The model is loaded again on the newly configured device (mps or cpu).

## 10. Dataset after Processing

- **Processed Dataset**:
  - `dataset_hidden_states`: Displays the dataset with the added `hidden_state` feature, which contains the extracted hidden states from the model.

## 11. Preparing Data for Machine Learning

- **Feature and Target Preparation**:
  - `X_train` and `X_valid`: Convert the hidden states from the training and validation datasets into NumPy arrays.
  - `y_train` and `y_valid`: Convert the labels from the training and validation datasets into NumPy arrays.

- **Shape of Data**:
  - `X_train.shape` and `X_valid.shape`: Print the shapes of the training and validation data, which helps verify that the data has been processed correctly.

## 12. Logistic Regression

- **Model Creation**:
  - `LogisticRegression(max_iter=3000)`: Initializes a logistic regression model with a maximum of 3000 iterations for convergence.

- **Training**:
  - `lr_clf.fit(X_train, y_train)`: Trains the logistic regression model on the training data.

- **Evaluation**:
  - `lr_clf.score(X_valid, y_valid)`: Computes the accuracy of the model on the validation set. The result is rounded to 3 decimal places, showing an accuracy of `0.4`.

## 13. Support Vector Machine (SVM)

- **Model Creation**:
  - `SVC()`: Initializes a support vector classifier. The default parameters are used.

- **Training**:
  - `svm_clf.fit(X_train, y_train)`: Trains the SVM model on the training data.

- **Evaluation**:
  - `svm_clf.score(X_valid, y_valid)`: Computes the accuracy of the SVM model on the validation set. The result is rounded to 3 decimal places, showing an accuracy of `0.389`.

## 14. Random Forest

- **Model Creation**:
  - `RandomForestClassifier()`: Initializes a random forest classifier with default parameters.

- **Training**:
  - `rf_clf.fit(X_train, y_train)`: Trains the random forest model on the training data.

- **Evaluation**:
  - `rf_clf.score(X_valid, y_valid)`: Computes the accuracy of the random forest model on the validation set. The result is rounded to 3 decimal places, showing an accuracy of `0.556`.

## 15. Transformer-Based Model

- **Model Creation**:
  - `AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)`: Initializes a transformer model for sequence classification using the pre-trained checkpoint `'ai4bharat/indic-bert'`. The model is set to handle 3 different labels.

- **Performance Metrics Function**:
  - `perf_metrics(pred)`: Defines a function to compute performance metrics for the model. It calculates:
    - **Accuracy**: The proportion of correct predictions.
    - **F1 Score**: A weighted average of precision and recall.

- **Training Arguments**:
  - `TrainingArguments`: Sets up the arguments for training the transformer model:
    - `output_dir`: Directory where model outputs will be saved.
    - `num_train_epochs`: Number of epochs for training.
    - `learning_rate`: Learning rate for the optimizer.
    - `per_device_train_batch_size` and `per_device_eval_batch_size`: Batch sizes for training and evaluation.
    - `weight_decay`: Regularization parameter.
    - `evaluation_strategy`: Determines how often to evaluate the model during training.
    - `logging_steps`: Defines the number of steps between logging updates.

- **Trainer Initialization**:
  - `Trainer`: Initializes the Trainer class from the Hugging Face transformers library with the following:
    - `model`: The model to be trained.
    - `args`: The training arguments.
    - `compute_metrics`: Function to compute metrics.
    - `train_dataset`: Training dataset.
    - `eval_dataset`: Evaluation dataset.
    - `tokenizer`: Tokenizer used for preprocessing text.

- **Training**:
  - `trainer.train()`: Starts the training process for the transformer model.

## 16. Prediction Output

- **Prediction**:
  - `preds_output = trainer.predict(dataset_encoded['test'])`: Obtains predictions from the trained model on the test dataset. The `predict` method of the `Trainer` class runs inference on the test data and returns predictions along with evaluation metrics.

- **Evaluation Metrics**:
  - **`test_loss`**: `1.0860443115234375` - The average loss value on the test dataset.
  - **`test_accuracy`**: `0.57777777777777777` - The accuracy of the model on the test set, approximately `58%`.
  - **`test_f1 score`**: `0.5138916042141849` - The F1 score, approximately `0.51`, suggesting some balance between precision and recall.
  - **`test_runtime`**: `11.0024` - The total time taken to evaluate the model on the test dataset, measured in seconds.
  - **`test_samples_per_second`**: `8.18` - The average number of test samples processed per second.
  - **`test_steps_per_second`**: `1.636` - The average number of steps (batches) processed per second.
