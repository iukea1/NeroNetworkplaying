import sklearn
import numpy as np
import pandas as pd
import torch 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset['data'], columns = boston_dataset['feature_names'])
boston['MEDV'] = boston_dataset['target']

X = pd.DataFrame(np.c_[boston['CRIM'],boston[ 'ZN'], boston[ 'INDUS' ], boston['CHAS'], boston[ 'NOX' ], boston['RM'], boston['AGE'], boston['DIS'], boston[ 'RAD'], boston['TAX' ], boston['PTRATIO'], boston['B'], boston['LSTAT']], columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
Y = boston['MEDV']
#prin test
print (X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 666)

torch_tensor = torch.Tensor(np.array(X_train))
#print(torch_tensor)
y = torch.Tensor(np.array(Y_test))
T = torch.Tensor(np.array(Y_train))
X_test = torch.Tensor(np.array(X_test))


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x) # instead of Heaviside step fn
        return output

class Perceptron2(torch.nn.Module):
    def __init__(self):
        super(Perceptron2, self).__init__()
        self.fc = torch.nn.Linear(1,1)
        self.Hardtanh = torch.nn.Hardtanh() # instead of Heaviside step fn
        print ("I made it to perception2")
    def forward(self, x):
        output = self.fc(x)
        output = self.Hardtanh(x) # instead of Heaviside step fn
        return output


class Feedforward(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, hidden_size2):
    
        super(Feedforward, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size2)
        self.Hardtanh = torch.nn.Hardtanh()

        self.fc3 = torch.nn.Linear(self.hidden_size2, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        
        #print ("made it into the second feedforword")
        
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        
        hidden2 = self.fc2(relu)
        Hardtanh = self.Hardtanh(hidden2)

        output = self.fc3(Hardtanh)
        output = self.sigmoid(output)
        
        return output

#model optimizaer
model = Feedforward(13, 16, 32)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


#trail the model
model.eval()
y_pred = model(X_test)
before_train = criterion(y_pred.squeeze(), y)
print('Test loss before training' , before_train.item())



#training!!!

model.train()
epoch = 100000
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(torch_tensor)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), T.squeeze())
   
    #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()

model.eval()
y_pred = model(X_test)
after_train = criterion(y_pred.squeeze(), y) 
print('Test loss after Training' , after_train.item())

