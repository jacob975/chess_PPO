import torch as th
import numpy as np
from gym import spaces
from custom_policy import SupervisedCNN
from torchsummary import summary
import os
import tqdm


# Load the model
# The model output is a 4672-dim sigmoid vector
model = SupervisedCNN(
    observation_space=spaces.Box(low=0, high=1, shape=(111, 8, 8), dtype=np.float32),
    features_dim=4672,
    normalize=False,
    activation_fn=th.nn.Sigmoid(),
)
# Activate cuda
model = model.cuda()
summary(model, (111, 8, 8))

# Load the data
print("Loading data...")
observations = np.load("datasets/observations.npy") # (8, 8, 111) binary vectors
action_masks = np.load("datasets/action_masks.npy") # (4672,) binary vectors

# Convert the data to tensors
observations = th.from_numpy(observations).float()
# Permute the observations to (batch_size, channels, height, width)
observations = observations.permute(0, 3, 1, 2)
action_masks = th.from_numpy(action_masks).float()
# Construct training and validation sets
train_size = int(0.8 * len(observations))
batch_size = 256
train_observations = observations[:train_size]
train_action_masks = action_masks[:train_size]
val_observations = observations[train_size:]
val_action_masks = action_masks[train_size:]

# Training
print("Training...")
model.train()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)
loss_fn = th.nn.BCELoss()
for epoch in range(10):
    # Use tqdm to show the progress bar with loss as the postfix
    loss_sum = 0.0
    pbar = tqdm.tqdm(range(0, train_size, batch_size), postfix={"loss": 0.0})
    for i in pbar:
        optimizer.zero_grad()
        batch_observations = train_observations[i:i+batch_size]
        batch_action_masks = train_action_masks[i:i+batch_size] # (batch_size, 4672)
        # Move the data to cuda
        batch_observations = batch_observations.cuda()
        batch_action_masks = batch_action_masks.cuda()
    
        outputs = model(batch_observations)
        loss = loss_fn(outputs, batch_action_masks)
        loss.backward()
        # Update the loss sum
        loss_sum += loss.item()
        pbar.set_postfix({"loss": loss_sum / (i + 1)})
        optimizer.step()

    # Validation
    loss_sum = 0.0
    with th.no_grad():
        for i in range(0, len(val_observations), batch_size):
            batch_observations = val_observations[i:i+batch_size]
            batch_action_masks = val_action_masks[i:i+batch_size]
            # Move the data to cuda
            batch_observations = batch_observations.cuda()
            batch_action_masks = batch_action_masks.cuda()
            outputs = model(batch_observations)
            loss = loss_fn(outputs, batch_action_masks)
            loss_sum += loss.item()
        loss = loss_sum / len(val_observations)
    # Report the loss
    print("Epoch: {}, Loss: {}".format(epoch, loss))

# Save the model
th.save(model.state_dict(), "models/supervised_model.pth")

