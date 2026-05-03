import numpy
import pandas as pd
import torch

from torch.utils.data import random_split, TensorDataset,WeightedRandomSampler
# from torchmetrics.classification import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from sklearn.model_selection import train_test_split



if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



class_names = ['Normal', 'Broken', 'Recovering']
# train_acc = Accuracy(task="multiclass", num_classes=3).to(device)
train_acc = MulticlassAccuracy(num_classes=3,average='none').to(device)
weights = torch.tensor([1.0, 200.0, 1.0]).to(device)

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
# X = torch.tensor(data[sensor_cols].values, dtype=torch.float32)
# y = torch.tensor(data["machine_status"].values, dtype=torch.long)

# dataset = TensorDataset(X, y)
normal_data = data[data["machine_status"] == 0]
anomalous_data = data[data["machine_status"] != 0]

train_normal_data, test_normal_data = train_test_split(
    normal_data,
    test_size=0.2,
    random_state=42,
)

test_df = pd.concat([test_normal_data, anomalous_data])
X_train = torch.tensor(train_normal_data[sensor_cols].values, dtype=torch.float32)
y_train = torch.tensor(train_normal_data["machine_status"].values, dtype=torch.long)

X_test = torch.tensor(test_df[sensor_cols].values, dtype=torch.float32)
y_test = torch.tensor(test_df["machine_status"].values, dtype=torch.long)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)


# sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=True
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=False
)
# Check how many 'Broken' samples made it into training
# train_broken_count = (y[train_idx] == 1).sum().item()
# print(f"Number of 'Broken' samples in Training Set: {train_broken_count}")
# print(data.head())
print(f"Using device: {device}")

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        # Encoder: Compresses the data
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 20),
            torch.nn.ReLU()
        )
        # Decoder: Attempts to reconstruct the input
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
model = NeuralNetwork(input_size=len(sensor_cols)).to(device)
# loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train (traindata,model ,loss_fn,optimizer):
    model.train()
    size = len(traindata.dataset)

    for batch, (X, _) in enumerate(traindata):
        # X, y = X.to(device), y.to(device)
        X = X.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def errorEvaluation(testdata, model, loss_fn):
    model.eval()
    train_errors = []
    # We track the reconstruction error for each category
    errors = {name: [] for name in class_names}
    
    with torch.no_grad():
        for X, y in testdata:
            X = X.to(device)
            # 1. Predict (Reconstruct)
            reconstructed = model(X)
            
            # 2. Calculate MSE loss per sample in the batch
            # reduction='none' gives us the loss for every individual row
            loss_per_sample = torch.nn.functional.mse_loss(reconstructed, X, reduction='none').mean(dim=1)
            train_errors.extend(loss_per_sample.cpu().numpy())
            # 3. Store errors based on their true labels
            for i, label in enumerate(y):
                class_name = class_names[label.item()]
                errors[class_name].append(loss_per_sample[i].item())

    mean_err = numpy.mean(train_errors)
    std_err = numpy.std(train_errors)
    threshold = mean_err + (3 * std_err)
    print("\n--- Calibration Results (Baseline) ---")
    print(f"Mean: {mean_err:.6f} | Std: {std_err:.6f}")
    print(f"Calculated Threshold: {threshold:.6f}")

    print("--- Avg Reconstruction Error (Lower is better) ---")
    for name in class_names:
        if errors[name]:
            avg_err = numpy.mean(errors[name])
            print(f"{name:12}: {avg_err:.6f}")
    print("--------------------------------------------------")

    return threshold

def test(threshold, test_dataloader, model):
    print(f"\n--- Detailed Performance Breakdown (Threshold: {threshold:.6f}) ---")
    model.eval()
    
    # Initialize counters for each class
    # Structure: { 'Normal': [count_below, count_above], ... }
    results = {name: {"below": 0, "above": 0} for name in class_names}
    
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            reconstructed = model(X_batch)
            
            # Calculate individual errors
            loss = torch.nn.functional.mse_loss(reconstructed, X_batch, reduction='none').mean(dim=1)
            
            # Check each sample in the batch
            for i in range(len(y_batch)):
                label_idx = y_batch[i].item()
                label_name = class_names[label_idx]
                
                if loss[i] > threshold:
                    results[label_name]["above"] += 1
                else:
                    results[label_name]["below"] += 1

    # Print formatting
    print(f"{'Class Name':<12} | {'Total':<8} | {'Predicted Normal':<18} | {'Predicted Anomaly'}")
    print("-" * 65)
    
    for name in class_names:
        below = results[name]["below"]
        above = results[name]["above"]
        total = below + above
        
        # Calculate success rate based on class type
        if name == 'Normal':
            success_rate = (below / total * 100) if total > 0 else 0
            note = "(Stayed Below)"
        else:
            success_rate = (above / total * 100) if total > 0 else 0
            note = "(Flagged Above)"
            
        print(f"{name:<12} | {total:<8} | {below:<18} | {above:<17} | {success_rate:>6.2f}% {note}")

    print("-" * 65)
epochs = 20
for t in range(epochs):
    print(f"-----------------Epoch {t+1}--------------")
    train(train_dataloader, model, loss_fn, optimizer)
   
threshold = errorEvaluation(train_dataloader, model, loss_fn)
test(threshold, test_dataloader, model)