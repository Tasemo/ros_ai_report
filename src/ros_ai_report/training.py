#!/usr/bin/env python3

import torch
import torchaudio
import pickle
import matplotlib.pyplot as plt
from dataset import DataSetType, DataSubSet
from model import Model

def collate_fn(batch):
    tensors, targets = [], []

    def pad_with_zero(batch):
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        return batch.permute(0, 2, 1)

    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [torch.tensor(labels.index(label))]
    return pad_with_zero(tensors), torch.stack(targets)

def trainModel():
    model.train()
    total_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(transform(data))
        loss = loss_fn(output.squeeze(), target)
        total_loss += loss.item()
        prediced = output.argmax(dim=-1).squeeze()
        correct += prediced.eq(target).sum().item()
        loss.backward()
        optimizer.step()
    data_size = len(train_loader.dataset)
    total_loss /= data_size
    train_losses.append(total_loss)
    percentage = 100. * correct / data_size
    train_accuracy.append(percentage)
    print("Training set: Average loss: {:.4f}, Accuracy: {}/{}, ({:.1f}%)".format(total_loss, correct, data_size, percentage))

def testModel():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(transform(data))
            total_loss += loss_fn(output.squeeze(), target).item()
            prediced = output.argmax(dim=-1).squeeze()
            correct += prediced.eq(target).sum().item()
    data_size = len(test_loader.dataset)
    total_loss /= data_size
    test_losses.append(total_loss)
    percentage = 100. * correct / data_size
    test_accuracy.append(percentage)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{}, ({:.1f}%)".format(total_loss, correct, data_size, percentage))

def savePlot():
    plt.xlabel("Epoch")
    plt.ylabel("Aerage Loss")
    plt.plot(train_losses, label="Training Set")
    plt.plot(test_losses, label="Test Set")
    plt.legend()
    plt.savefig("model/model_losses.png")
    plt.close()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracy, label="Training Set")
    plt.plot(test_accuracy, label="Test Set")
    plt.legend()
    plt.savefig("model/model_accuracies.png")
    plt.close()

def saveMetadata():
    metadata = {}
    metadata["sample_rate"] = sample_rate
    metadata["labels"] = labels
    with open("model/metadata", "wb") as file:
        pickle.dump(metadata, file)
    
if __name__ == "__main__":
    sample_rate = 8000
    batch_size = 512
    num_workers = 4
    epochs = 6
    train_set = DataSubSet(DataSetType.TRAINING)
    test_set = DataSubSet(DataSetType.TESTING)

    labels = train_set.get_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchaudio.transforms.Resample(orig_freq=train_set.get_sample_rate(), new_freq=sample_rate).to(device)

    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    model = Model(output_size=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = torch.nn.NLLLoss()
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(1, epochs + 1):
        trainModel()
        testModel()
    torch.save(model.state_dict(), "model/trainedModel.pt")
    saveMetadata()
    savePlot()
