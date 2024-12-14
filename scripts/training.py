import warnings
from model import *

warnings.filterwarnings("ignore")


def train(data):
    # Enumerate over the data
    for batch in loader:
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        # Update using the gradients
        optimizer.step()
    return loss, embedding


model = GCN()

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(
    data[: int(data_size * 0.8)], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True
)  # 20% training

test_loader = DataLoader(
    data[int(data_size * 0.8) :], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True
)  # 80% testing

print("Start training...")
losses = []
for epoch in range(2000):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Train Loss {loss}")
