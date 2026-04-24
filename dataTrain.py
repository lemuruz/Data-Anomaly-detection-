import pandas as pd
import torch
from torch.utils.data import random_split, TensorDataset
data = pd.read_csv('sensor.csv')
data["machine_status"] = (
    data["machine_status"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        'NORMAL': 0,
        'BROKEN': 1,
        'RECOVERING': 2
    })
)

data = data.dropna(subset=['machine_status'])
data["machine_status"] = data["machine_status"].astype(int)

# sensor preprocessing
sensor_cols = [col for col in data.columns if col.startswith("sensor_")]

data[sensor_cols] = data[sensor_cols].ffill().bfill()
std = data[sensor_cols].std()
std[std == 0] = 1
data[sensor_cols] = (data[sensor_cols] - data[sensor_cols].mean()) / std
data[sensor_cols] = data[sensor_cols].replace([float("inf"), -float("inf")], 0).fillna(0)
# convert data
X = torch.tensor(data[sensor_cols].values, dtype=torch.float32)
y = torch.tensor(data["machine_status"].values, dtype=torch.long)
print("X has NaN:", torch.isnan(X).any().item())
print("X has Inf:", torch.isinf(X).any().item())
print("y has NaN:", torch.isnan(y).any().item())
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# for col in data.columns:
#     if col.startswith("sensor_"):
#         data[col] = data[col].fillna(data[col].mean())
#         std = data[col].std()
#         if std != 0:
#             data[col] = (data[col] - data[col].mean()) / std
print(data.head())
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        return out
model = NeuralNetwork(input_size=len(sensor_cols), hidden_size=64, output_size=3).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

def train (traindata,model ,loss_fn,optimizer):
    model.train()
    size = len(traindata.dataset)

    for batch, (X, y) in enumerate(traindata):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(testdata, model, loss_fn):
    size = len(testdata.dataset)
    num_batches = len(testdata)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in testdata:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")