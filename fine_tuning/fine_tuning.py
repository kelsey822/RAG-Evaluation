"""Script to fine tune model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import torch
from google.colab import drive

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_file = '/content/drive/MyDrive/embeddings/responses.csv'
model_name = 'all-MiniLM-L6-v2'
epochs = 3
batch_size = 16
output_dir = '/content/drive/MyDrive/embeddings/fine_tuned_model'


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'question'}, inplace=True)
        df.to_csv(file_path, index=False)
    return df

def prepare_training_data(df):
    return [
        InputExample(texts=[str(row['question']).lower().strip(),
                            str(row[f'source{i}']).lower().strip()],
                     label=float(row[f'relevance{i}']) / 4.0)
        for _, row in df.iterrows()
        for i in range(1, 6)
        if not pd.isna(row[f'relevance{i}']) and row[f'relevance{i}'] != -1
    ]

def fine_tune_model(training_examples, model_name='all-MiniLM-L6-v2', epochs=3, batch_size=16):
    model = SentenceTransformer(model_name)
    model = model.to(device)
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100, show_progress_bar=True)
    return model

def evaluate_model(model, test_examples):
    true_scores, pred_scores = [], []
    for example in tqdm(test_examples):
        true_score = example.label
        embeddings = model.encode(example.texts, device=device)
        pred_score = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        true_scores.append(true_score)
        pred_scores.append(pred_score)

    true_scores, pred_scores = np.array(true_scores), np.array(pred_scores)
    correlation, p_value = spearmanr(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)
    return correlation, mse, true_scores, pred_scores


df = load_and_preprocess_data(data_file)
training_examples = prepare_training_data(df)
train_examples, test_examples = train_test_split(training_examples, test_size=0.2, random_state=42)

model = fine_tune_model(train_examples, model_name=model_name, epochs=epochs, batch_size=batch_size)

os.makedirs(output_dir, exist_ok=True)
model.save(os.path.join(output_dir, 'fine_tuned_model'))


base_model = SentenceTransformer(model_name)
fine_tuned_model = SentenceTransformer(os.path.join(output_dir, 'fine_tuned_model'))

base_corr, base_mse, base_true, base_pred = evaluate_model(base_model, test_examples)
ft_corr, ft_mse, ft_true, ft_pred = evaluate_model(fine_tuned_model, test_examples)

print(f"Base Model - Correlation: {base_corr:.4f}, MSE: {base_mse:.4f}")
print(f"Fine-tuned Model - Correlation: {ft_corr:.4f}, MSE: {ft_mse:.4f}")

# Cell 7: Qualitative Analysis
for i in range(min(5, len(test_examples))):
    example = test_examples[i]
    print(f"\nQuestion: {example.texts[0]}")
    print(f"Source: {example.texts[1][:100]}...")
    print(f"True Score: {example.label:.4f}")
    print(f"Base Model Prediction: {base_pred[i]:.4f}")
    print(f"Fine-tuned Model Prediction: {ft_pred[i]:.4f}")



