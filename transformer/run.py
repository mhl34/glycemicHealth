import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from PositionalEncoding import PositionalEncoding
from hyperParams import hyperParams
from DataProcessor import DataProcessor
from pp5 import pp5
from Transformer import Transformer
from utils import createPersSamples

class runModel:
    def __init__(self, mainDir):
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.sequence_length = 128

    def run(self):
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        hyperparams = hyperParams()
        pp5vals = pp5()

        # get eda, temp, glucose, and hr data of all the samples
        samples = [str(i).zfill(3) for i in range(1, 17)]
        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")
        # 
        X, y = createPersSamples(glucoseData, edaData, hrData, tempData, pp5vals, 12, 1000, samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        train_data = [[X_train[i], y_train[i]] for i in range(len(X_train))]
        test_data = [[X_test[i], y_test[i]] for i in range(len(X_test))]
        train_dataloader = self.createBatches(train_data, batch_size = 16)
        test_dataloader = self.createBatches(test_data, batch_size = 16)

        model = Transformer(
            num_tokens=hyperparams.NUM_TOKENS,
            embedding_dim_encode=hyperparams.EMBEDDING_DIM_ENCODE,
            embedding_dim_decode=hyperparams.EMBEDDING_DIM_DECODE,
            num_heads=hyperparams.NUM_HEADS,
            num_encoder_layers=hyperparams.NUM_ENCODER_LAYERS,
            num_decoder_layers=hyperparams.NUM_DECODER_LAYERS,
            dropout_p=hyperparams.DROPOUT_P,
            norm_first=hyperparams.NORM_FIRST,
            device = self.device
        ).to(self.device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
        opt = optim.SGD(model.parameters(), lr=hyperparams.LEARNING_RATE, momentum = 0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)
        CHECKPOINT_FOLDER = "./saved_model"

        train_loss_list, validation_loss_list = self.fit(model, opt, criterion, train_dataloader, test_dataloader, 20)

    def createBatches(self, data, batch_size=16, padding=False, padding_token=-1):
        batches = []
        idx = 0
        while idx + batch_size < len(data):
            batches.append(np.array(data[idx : idx + batch_size]))
            idx += batch_size
        print(f"{len(batches)} batches of size {batch_size}")
        return batches

    def train(self, model, opt, criterion, dataloader):
        model.train()
        total_loss = 0

        accuracy = []

        for batch in dataloader:
            # X --> first half
            # y --> second half
            X, y = batch[:, 0], batch[:, 1]
            X = [torch.tensor(element) for element in X]
            y = [torch.tensor(element) for element in y]
            X, y = torch.stack(X).to(self.device), torch.stack(y).to(self.device)
            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            # y_input [SOS_token, ...]
            y_input = y[:-1]
            # y_expected [..., EOS_token]
            y_expected = y[1:]

            # Get mask to mask out the next words
            tgt_mask = model.get_tgt_mask(self.sequence_length).to(self.device)
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0).float()
            # argmax gives the prediction
            y_pred = torch.argmax(pred.detach(), axis=1).float()
            y_pred.requires_grad = True
            y_expected = y_expected.float()
            y_expected.requires_grad = True
            # print(torch.argmax(pred.detach(), axis=1))
            mape = self.mean_absolute_percentage_error(y_expected, y_pred)
            accuracy.append(100 - mape.detach().item())
            # accuracy.append(comp_accuracy(y_pred, y_expected).item())
            loss = criterion(y_pred, y_expected)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = np.mean(accuracy)

        return epoch_loss, epoch_acc


    def evaluate(self, model, criterion, dataloader):
        model.eval()
        total_loss = 0

        accuracy = []

        with torch.no_grad():
            for batch in dataloader:
                # X --> first half
                # y --> second half
                X, y = batch[:, 0], batch[:, 1]
                X = [torch.tensor(element) for element in X]
                y = [torch.tensor(element) for element in y]
                X, y = torch.stack(X).to(self.device), torch.stack(y).to(self.device)
                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                # y_input [SOS_token, ...]
                y_input = y[:-1]
                # y_expected [..., EOS_token]
                y_expected = y[1:]

                # Get mask to mask out the next words
                tgt_mask = model.get_tgt_mask(self.sequence_length).to(self.device)
                pred = model(X, y_input, tgt_mask)

                # Permute pred to have batch size first again
                pred = pred.permute(1, 2, 0).float()
                # argmax gives the prediction
                y_pred = torch.argmax(pred.detach(), axis=1).float()
                y_pred.requires_grad = True
                y_expected = y_expected.float()
                y_expected.requires_grad = True
                mape = self.mean_absolute_percentage_error(y_expected, y_pred)
                accuracy.append(100 - mape.cpu().detach().item())
                # accuracy.append(comp_accuracy(y_pred, y_expected).item())
                loss = criterion(y_pred, y_expected)
                
                total_loss += loss.detach().item()

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = np.mean(accuracy)

        return epoch_loss, epoch_acc

    def fit(self, model, opt, criterion, train_dataloader, test_dataloader, epochs):
        best_test_acc = float('-inf')
        print("Training and validating model")
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")

            train_loss, train_accuracy = self.train(model, opt, criterion, train_dataloader)
            print(f"Training loss: {train_loss:.4f}")
            print(f"Training accuracy: {train_accuracy:.4f}")
            # print(f"Validation loss: {validation_loss:.4f}")
            # print(f"Validation accuracy: {validation_accuracy:.4f}")
            if train_loss < best_test_acc:
                best_test_acc = train_loss
                if not os.path.exists(CHECKPOINT_FOLDER):
                    os.makedirs(CHECKPOINT_FOLDER)
                print("Saving ...")
                state = {'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'lr': 0.001}
                torch.save(state['state_dict'], os.path.join(CHECKPOINT_FOLDER, 'ermTransformerPersGluc.pth'))

            test_loss, test_accuracy = self.evaluate(model, criterion, test_dataloader)
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy

        print(f"Test accuracy after {epochs} epochs: {best_test_acc}")

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred))/300)/100


if __name__ == "__main__":
    mainDir = "/home/mhl34/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()