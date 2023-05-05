# Eugenio Culurciello
# learn sequence code LSTM
x
import torch
import numpy as np
import matplotlib.pyplot as plt


class myDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.inputs = data[:,:-1]
        self.labels = data[:,1:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        X = self.inputs[idx]
        Y = self.labels[idx]
        sample = (X,Y)
        return sample


class ourModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(128, 1)
    def forward(self, x, h0, c0):
        x, (hn,cn) = self.lstm(x,(h0,c0))
        x = self.linear(x)
        return x, (hn,cn)


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        h0 = torch.zeros(2, 1, 128)
        c0 = torch.zeros(2, 1, 128)
        output,_ = model(data, h0, c0)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    # correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            h0 = torch.zeros(2, data.shape[0], 128)
            c0 = torch.zeros(2, data.shape[0], 128)
            output,_ = model(data, h0, c0)
            test_loss += torch.nn.functional.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    return test_loss


# main
train_steps = 15
np.random.seed(0)
torch.manual_seed(0)

# test data:
data_x_axis = np.linspace(-2*np.pi, 2*np.pi, 1000)
timeseries = torch.Tensor(np.sin(data_x_axis)).unsqueeze(0).unsqueeze(2) # batch and feature size = 1
# timeseries = timeseries.reshape(1,1,-1)
timeseries = (timeseries - timeseries.mean())/timeseries.std() # normalize
print('timeseries input shape:', timeseries.shape)

train_size = int(timeseries.shape[1] * 0.9)
train_data = timeseries[:,:train_size]
test_data  = timeseries[:,train_size:]
print('train data size:', train_data.shape)
print('test data size:', test_data.shape)

# make train, test datasets
train_dataset = myDataset(train_data)
test_dataset = myDataset(test_data)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=8)

# model
model = ourModel()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()
 
# train
n_epochs = 100
for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer, epoch)
    test_loss = test(model, test_loader)
    if epoch % 10 == 0:
        print('Train Epoch:', epoch, 'Train Loss:', f"{train_loss:.4f}", 
              'Test loss:', f"{test_loss:.4f}")

# torch.save(model.state_dict(), 'trained_model.pth')

# plot predictions
original_data_plot = timeseries.reshape(-1)
pred_plot_data = np.nan * original_data_plot
stdsh = int(timeseries.shape[1]/2) # index at which to start predicting
with torch.no_grad():
    h0 = torch.zeros(2, 1, 128)
    c0 = torch.zeros(2, 1, 128)
    train_data_plot,_ = model(train_data[:,:stdsh], h0, c0)
    train_data_plot = train_data_plot.reshape(-1)
    inputs = train_data[:,:1,:] # first input
    hn = torch.zeros(2, 1, 128)
    cn = torch.randn(2, 1, 128)
    for t in range(stdsh):
        inputs, (hn,cn) = model(inputs, hn, cn)
        pred_plot_data[stdsh+t] = inputs.squeeze(0).squeeze(1)

plt.plot(original_data_plot, c='black')
plt.plot(train_data_plot, c='red')
plt.plot(pred_plot_data, c='green')
plt.show()
