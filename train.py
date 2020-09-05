import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_utils import get_train_test_loaders
from model import GeneClassifier


BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epoch, model, train_loader, criterion, optimizer, writer=None, log_every=10):
    print("Epoch:", epoch)
    model.train()

    running_loss = 0
    average_losses = []

    start_time_s = time.time()

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        ids, masks, variants = batch
        
        ids = ids.to(device)
        masks = masks.to(device)
        variants = variants.to(device)

        predictions = model(ids, masks)

        loss = criterion(predictions, variants.flatten())
        loss.backward()

        running_loss += loss.item()

        optimizer.step()

        if (i + 1) % log_every == 0:
            sec_per_it = round((time.time() - start_time_s) / log_every, 3)
            average_loss = round(running_loss / log_every, 4)
            number_of_steps = epoch * len(train_loader) + i

            print(f"\tIteration = {i + 1}\tLoss = {average_loss}\t{sec_per_it}s/it")

            if writer:
                writer.add_scalar("train_loss", average_loss, number_of_steps)
                writer.close()
            
            running_loss = 0
            start_time_s = time.time()


def test(epoch, model, test_loader, criterion, writer=None, log_every=10):
    print("Testing")
    model.eval()

    min_average_loss = float("inf")
    total_correct = 0
    running_correct = 0
    running_loss = 0

    start_time_s = time.time()

    for i, batch in enumerate(test_loader):
        ids, masks, variants = batch
        
        ids = ids.to(device)
        masks = masks.to(device)
        variants = variants.to(device)

        predictions = model(ids, masks)

        loss = criterion(predictions, variants.flatten())
        running_loss += loss.item()

        correct = (torch.max(predictions, 1).indices == variants.flatten()).sum().item()
        total_correct += correct
        running_correct += correct

        if (i + 1) % log_every == 0:
            sec_per_it = round((time.time() - start_time_s) / log_every, 3)
            average_loss = round(running_loss / log_every, 4)
            average_accuracy = round(running_correct / (log_every * ids.size(0)), 5) * 100
            number_of_steps = epoch * len(test_loader) + i

            print(f"\tIteration = {i + 1}\tLoss = {average_loss}\tAcccuracy = {average_accuracy}%\t{sec_per_it}s/it")

            if average_loss < min_average_loss:
                min_average_loss = average_loss
                torch.save(model.state_dict(), f"models/model_{epoch}.pth")

            if writer:
                writer.add_scalar("test_loss", average_loss, number_of_steps)
                writer.add_scalar("test_accuracy", average_accuracy, number_of_steps)
                writer.close()
            
            running_loss = 0
            running_correct = 0
            start_time_s = time.time()

    print(f"\tAccuracy: {round(total_correct / len(test_loader) / ids.size(0), 5) * 100}%")


def main(batch_size=BATCH_SIZE):
    if not os.path.exists("models/"):
        os.makedirs("models/")

    writer = SummaryWriter()

    train_loader, test_loader = get_train_test_loaders(batch_size)

    model = GeneClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train(epoch, model, train_loader, criterion, optimizer, writer)
        test(epoch, model, test_loader, criterion, writer)


if __name__ == "__main__":
    main()