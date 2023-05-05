# Eugenio Culurciello
# learn sequence code - Unet sequence to sequence S2S

import torch
import numpy as np
import matplotlib.pyplot as plt


class s2sDataset(torch.utils.data.Dataset):
    def __init__(self, S1, S2):
        self.inputs = S1
        self.labels = S2

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        X = self.inputs[idx]
        Y = self.labels[idx]
        sample = (X,Y)
        return sample

# model for sequence to sequence with u-net like model
class unetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv1d(1, 16, 32, stride=8)
        self.c2 = torch.nn.Conv1d(16, 32, 16, stride=4)
        self.c3 = torch.nn.Conv1d(32, 64, 8, stride=2)
        self.u3 = torch.nn.ConvTranspose1d(64, 32, 8, stride=2)
        self.u2 = torch.nn.ConvTranspose1d(32, 16, 16, stride=4, output_padding=1)
        self.u1 = torch.nn.ConvTranspose1d(16, 1, 32, stride=8, output_padding=4)

    def forward(self, x):
        # print(x.shape)
        x = self.c1(x)
        # print(x.shape)
        x = self.c2(x)
        # print(x.shape)
        x = self.c3(x)
        # print(x.shape)
        x = self.u3(x)
        # print(x.shape)
        x = self.u2(x)
        # print(x.shape)
        x = self.u1(x)
        # print(x.shape)
        return x


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += torch.nn.functional.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    return test_loss


# main
train_steps = 15
np.random.seed(0)
torch.manual_seed(0)

# sequence 1 and 2:
data_x_axis = np.linspace(-2*np.pi, 2*np.pi, 1000)
S1 = torch.Tensor(np.sin(data_x_axis)).unsqueeze(0) # batch = 1
S1 = (S1 - S1.mean())/S1.std() # normalize
S2 = S1.pow(2)
S2 = (S2 - S2.mean())/S2.std() # normalize
print('S1/S2 shape:', S1.shape)

# plot debug:
# plt.plot(S1.reshape(-1), c='blue')
# plt.plot(S2.reshape(-1), c='red')
# plt.show()

train_size = int(S1.shape[1] * 0.9)
train_data_S1 = S1[:,:train_size]
train_data_S2 = S2[:,:train_size]
test_data_S1  = S1[:,train_size:]
test_data_S2  = S2[:,train_size:]
print('train data S1/S2 size:', train_data_S1.shape)
print('test data S1/S2 size:', test_data_S1.shape)

# make train, test datasets
train_dataset = s2sDataset(train_data_S1, train_data_S2)
test_dataset = s2sDataset(test_data_S1, test_data_S2)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=8)

# model
model = unetModel()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

# test model:
o = model(train_data_S1)
 
# train
n_epochs = 1000
for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer)
    test_loss = 0#test(model, test_loader)
    if epoch % 100 == 0:
        print('Train Epoch:', epoch, 'Train Loss:', f"{train_loss:.4f}", 
              'Test loss:', f"{test_loss:.4f}")

# torch.save(model.state_dict(), 'trained_model.pth')

# plot predictions
with torch.no_grad():
    pred_S2 = model(train_data_S1)

print(pred_S2.shape, pred_S2.mean(), pred_S2.std())

plt.plot(S1.reshape(-1), c='black')
plt.plot(S2.reshape(-1), c='red')
plt.plot(pred_S2.reshape(-1), c='green')
plt.show()
