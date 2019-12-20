import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)
    
batch_size = 100
learning_rate = 0.001
training_epochs = 15

pre_process = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

mnist_train = dsets.MNIST(root='PATH',
                          train=True, 
                          transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='PATH', 
                         train=False, 
                         transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size,
                                         shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep_prob = 0.5
        
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, padding=1))
        
        self.fc1 = nn.Linear(4*4*128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(self.fc1,
                                   nn.ReLU(),
                                   nn.Dropout(p=1-self.keep_prob))
        
        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)

#training
for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        prediction = model(X)
        optimizer.zero_grad()
        cost = criterion(prediction, Y)
        print(Y)
        print(prediction)
        print(cost)
        cost.backward()
        optimizer.step()
        
        avg_cost += (cost / total_batch)
        
    print('epoch : {}, cost : {}'.format(epoch, cost))
print('Finished!')

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('accuracy : ', accuracy.item())