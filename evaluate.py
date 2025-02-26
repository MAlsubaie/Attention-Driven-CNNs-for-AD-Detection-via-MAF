import numpy as np
import pandas as pd
import tensorflow as tf
from utils.dataloader import CustomDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from model.model import create_model
from tqdm import tqdm

def evaluate(args):
    try:
        model = create_model(input_shape=args.input_shape, load_pretrained=True, weights_path=args.weights_path)
        df_test = pd.read_csv(args.dataset_path + "test.csv")

        test_gen = CustomDataGenerator(df_test, batch_size=args.batch_size)
        y_true, y_pred = [], []
        for x, y in tqdm(test_gen):
            pred = model.predict(x, verbose=False).flatten()
            y_true.extend(y)
            y_pred.extend((pred > 0.5).astype(int))

        print(classification_report(y_true, y_pred, target_names=['CN', 'AD']))

        plt.figure(figsize=(10,8))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Greens', xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
        plt.savefig(args.save_CM_dir + 'confusion_matrix.png')

    except Exception as e:
        print("Error: ", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CNN model")
    parser.add_argument("--input_shape", type=tuple, default=(128, 128, 128, 1))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset_path", type=str, default="./dataset/")
    parser.add_argument("--weights_path", type=str, default="./model/weights/best_weights.keras")
    parser.add_argument("--save_CM_dir", type=str, default="./results/")
    args = parser.parse_args()

    evaluate(args)
