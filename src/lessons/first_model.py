import torch
import torch.nn as nn


# 1. Model
class OWMFirstModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 2. Training data
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
])


# 3. Create model
model = OWMFirstModel()


# 4. Loss function
loss_fn = nn.MSELoss()


# 5. Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# 6. Training
for step in range(1000):

    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(
            f"Step: {step}, "
            f"Loss: {loss.item():.6f}"
        )


# 7. Test
test = torch.tensor([[10.0]])

result = model(test)

print("Prediction for 10:", result.item())