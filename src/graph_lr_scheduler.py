import torch as T
import matplotlib.pyplot as plt

EPOCHS = 10
BATCHES = 10
steps = []
lrs = []
model = ...  # Your model instance
optimizer = T.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # Wrapped optimizer
scheduler = T.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.9, total_steps=EPOCHS * BATCHES
)
for epoch in range(EPOCHS):
    for batch in range(BATCHES):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch * BATCHES + batch)

plt.figure()
plt.legend()
plt.plot(steps, lrs, label="OneCycle")
plt.show()
