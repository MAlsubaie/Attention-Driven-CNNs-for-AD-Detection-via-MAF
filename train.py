import os
import pandas as pd
import tensorflow as tf
from utils.utility_fun import save_plot
from tensorflow.keras.optimizers import SGD
from utils.dataloader import CustomDataGenerator
from utils.callbacks import get_callbacks
from model.model import create_model
import argparse

def train_model(args):
    try:
        model = create_model(input_shape=args.input_shape, load_pretrained=args.load_pretrained)

        print("Model created successfully")

        # Compile the model
        print("Compiling the model")
        model.compile(
            optimizer=SGD(learning_rate=args.learning_rate, momentum=args.momentum),
            loss=tf.keras.losses.BinaryCrossentropy(name='loss'),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='acc'), 
                tf.keras.metrics.AUC(name='auc'),
                tf.metrics.Precision(name="precision"),
                tf.metrics.Recall(name="recall")
            ]
        )

        df_train = pd.read_csv(args.dataset_path + "train.csv")
        df_val = pd.read_csv(args.dataset_path + "val.csv")

        train_gen = CustomDataGenerator(df_train, batch_size=args.batch_size)
        val_gen = CustomDataGenerator(df_val, batch_size=args.batch_size)

        history = model.fit(
                train_gen, 
                validation_data=val_gen, 
                epochs=args.epochs, 
                callbacks=get_callbacks(args.save_weights_path, args.RLR_patience)
                )

        save_plot(history.history, 'loss', os.path.join(args.save_curves_dir, 'loss.png'))
        save_plot(history.history, 'acc', os.path.join(args.save_curves_dir, 'acc.png'))
        save_plot(history.history, 'auc', os.path.join(args.save_curves_dir, 'auc.png'))
        save_plot(history.history, 'precision', os.path.join(args.save_curves_dir, 'precision.png'))
        save_plot(history.history, 'recall', os.path.join(args.save_curves_dir, 'recall.png'))
    
    except Exception as e:
        print("Error: ", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CNN model")
    parser.add_argument("--input_shape", type=tuple, default=(128, 128, 128, 1))
    parser.add_argument("--load_pretrained", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="./dataset/")
    parser.add_argument("--save_weights_path", type=str, default="./model/weights/new_trained.keras")
    parser.add_argument("--RLR_patience", type=int, default=2)
    parser.add_argument("--save_curves_dir", type=str, default="./results/")
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_plot", type=bool, default=True)
    args = parser.parse_args()

    train_model(args)
