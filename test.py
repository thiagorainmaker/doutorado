import torch
from torchvision import datasets, models, transforms
from torch import nn
import pandas as pd
import numpy as np
import config
from tqdm import tqdm



def make_prediction(model, loader, output_csv="pred.csv"):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions

            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5) & (predictions < 5)] = 4

            predictions = torch.max(predictions.long(), dim=1)[0]
            preds.append(predictions.cpu().numpy())
            filenames += files



    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print("Done with predictions")


def check_accuracy(loader, model, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            predictions = model(x)



        # Convert MSE floats to integer predictions

        predictions[predictions < 0.5] = 0
        predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
        predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
        predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
        predictions[(predictions >= 3.5) & (predictions < 5.0)] = 4



        y = y.view(-1)

        predictions = torch.max(predictions.long(), dim=1)[0]


        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

        # add to lists
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
        all_labels, axis=0, dtype=np.int64
    )