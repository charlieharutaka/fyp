import time

import numpy as np
import torch
from torchaudio.transforms import MuLawEncoding


def eval_conditional_wavenet(model, loader_val, criterion, encoder=MuLawEncoding(), device=torch.device("cpu")):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader_val:
            inputs, targets, condition, _, _, _ = batch
            batch_size, _, timesteps = inputs.shape
            # inputs (B, 1, T) | targets (B, 1, T) | condition (B, mels, T + 1)
            # We need to reshape the tensors into:
            # inputs (B, 1, T) | targets (B, T) | condition (B, mels, T)
            targets = encoder(targets.squeeze(1))
            condition = condition.narrow(2, 1, timesteps)
            # Transfer them to the device
            inputs = inputs.to(device)
            targets = targets.to(device)
            condition = condition.to(device)

            # Get prediction
            preds = model(inputs, condition)
            # Calculate loss
            loss = criterion(preds, targets)
            losses.append(loss.item())
    model.train()
    return np.average(losses)


def train_conditional_wavenet(model,
                              optimizer,
                              criterion,
                              epochs,
                              loader_train,
                              loader_val=None,
                              encoder=MuLawEncoding(),
                              epochs_start=0,
                              counter_start=1,
                              print_every=1000,
                              save_every=10000,
                              validate_every=1000,
                              save_as=None,
                              writer=None,
                              device=torch.device("cpu")):
    # Count from 1
    counter = counter_start
    train_losses = []
    validation_scores = []
    running_loss = 0.0
    last_validation_score = None

    for epoch in range(epochs_start, epochs_start + epochs):
        model.train()
        for t, batch in enumerate(loader_train):
            optimizer.zero_grad()

            inputs, targets, condition, _, _, _ = batch
            batch_size, _, timesteps = inputs.shape
            # inputs (B, 1, T) | targets (B, 1, T) | condition (B, mels, T + 1)
            # We need to reshape the tensors into:
            # inputs (B, 1, T) | targets (B, T) | condition (B, mels, T)
            targets = encoder(targets.squeeze(1))
            condition = condition.narrow(2, 1, timesteps)
            # Transfer them to the device
            inputs = inputs.to(device)
            targets = targets.to(device)
            condition = condition.to(device)

            # Get prediction
            preds = model(inputs, condition)
            # Calculate loss
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # Validate if we need to
            if loader_val is not None and validate_every > 0 and counter % validate_every == 0:
                last_validation_score = eval_conditional_wavenet(
                    model, loader_val, criterion, encoder, device)
                validation_scores.append(last_validation_score)
                if writer is not None:
                    writer.add_scalar("Validation Score",
                                      last_validation_score, counter - 1)

            # Stats
            loss_item = loss.item()
            train_losses.append(loss_item)
            running_loss += loss_item
            if writer is not None:
                writer.add_scalar("Loss", loss_item, counter - 1)

            if print_every > 0 and counter % print_every == 0:
                current_time = time.strftime("%y/%m/%d %H:%M:%S")
                print(
                    f'[{current_time}] Epoch: {epoch+1} | Batch: {t+1} | Loss: {running_loss / print_every} | Last validation score: {last_validation_score}')
                running_loss = 0.0
                if writer is not None:
                    writer.flush()
            if save_as is not None and save_every > 0 and counter % save_every == 0:
                torch.save(model.state_dict(),
                           f'{save_as}.step_{counter}.pt')

            counter += 1

    current_time = time.strftime("%y/%m/%d %H:%M:%S")
    print(f"[{current_time}] Finish training after {counter} steps")
    return train_losses, validation_scores


# def train_tacotron(model,
#                    optimizer,
#                    criterion,
#                    epochs,
#                    loader_train,
#                 #    loader_val=None,
#                    epochs_start=0,
#                    print_every=1000,
#                    save_every=10000,
#                    validate_every=1000,
#                    save_as=None,
#                    writer=None,
#                    device=torch.device("cpu")):
#     # Count from 1
#     counter = 1
#     train_losses = []
#     running_loss = 0.0

#     for epoch in range(epochs_start, epochs_start + epochs):
#         model.train()
#         for t, batch in enumerate(loader_train):

