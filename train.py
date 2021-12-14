
import config
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch import nn
import torch.optim as optim
import torch
from torch.utils.data import ConcatDataset
import test

import time



from dataset import  DRDataset, train_transforms, test_transforms


train_ds = DRDataset(
        images_folder="datasets/IDRI/512/train",
        path_to_csv="datasets/IDRI/train_dr.csv",
        transform=train_transforms
    )

val_ds = DRDataset(
        images_folder="datasets/IDRI/512/train",
        path_to_csv="datasets/IDRI/valid_dr.csv",
        transform=test_transforms
    )

test_ds = DRDataset(
        images_folder="datasets/IDRI/512/test",
        path_to_csv="datasets/IDRI/test_dr.csv",
        transform=test_transforms
    )

#kaggle
train_ds_kaggle = DRDataset(
        images_folder="datasets/kaggle/512/train",
        path_to_csv="datasets/kaggle/train.csv",
        transform=train_transforms
    )


valid_ds_kaggle = DRDataset(
        images_folder="datasets/kaggle/512/train",
        path_to_csv="datasets/kaggle/valid.csv",
        transform=train_transforms
    )

test_ds_kaggle = DRDataset(
        images_folder="datasets/kaggle/512/test",
        path_to_csv="datasets/kaggle/test.csv",
        transform=train_transforms
    )


trains_ds = [test_ds_kaggle, train_ds_kaggle, train_ds]
datasets = ConcatDataset(trains_ds)


valid_ds = [val_ds, valid_ds_kaggle]
datasets_valid = ConcatDataset(valid_ds)

train_loader = DataLoader(
        datasets,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )

val_loader = DataLoader(
        datasets_valid,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )

test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

model = models.inception_v3(pretrained=True)

for param in model.parameters():
  param.requires_grad = False


fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5),
    nn.LogSoftmax(dim=1)
)

model = model.to(config.DEVICE)

loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

train_data_size = len(train_ds)+len(test_ds_kaggle)+len(train_ds_kaggle)
valid_data_size = len(val_ds)+len(valid_ds_kaggle)
test_data_size = len(test_ds)


def train_and_validate(model, loss_criterion, optimizer, train_data_loader, valid_data_loader, epochs=25):
    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        # for i, (inputs, labels) in enumerate(train_data_loader):
        for batch_idx, sample in enumerate(train_data_loader):
            inputs = sample[0].to(config.DEVICE)
            labels = sample[1].to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs[0], 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(batch_idx, loss.item(),
                                                                                          acc.item()))

        # Validation
        with torch.no_grad():
            model.eval()
            # for j, (inputs, labels) in enumerate(valid_data_loader):
            for batch_idx, sample in enumerate(valid_data_loader):
                # inputs = inputs.to(DEVICE)
                # labels = labels.to(DEVICE)

                inputs = sample[0].to(config.DEVICE)
                labels = sample[1].to(config.DEVICE)

                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        # torch.save(model, 'model_'+str(epoch)+'.pt')

    return model, history, best_epoch


#

trained_model, history, best_epoch = train_and_validate(model, loss_func, optimizer, train_loader, val_loader, config.NUM_EPOCHS)
torch.save(history, 'models/model_history.pt')
torch.save(trained_model.state_dict(), 'models/trained_model.pt')


model = models.inception_v3(pretrained=True)

for param in model.parameters():
  param.requires_grad = False


fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5),
    nn.LogSoftmax(dim=1)
)

PATH = "models/trained_model.pt"
path_loader = torch.load(PATH)

model.load_state_dict(path_loader)
model.eval()
model.cuda()

test.make_prediction(model, test_loader, "models/pred.csv")
test.check_accuracy(test_loader, model)
