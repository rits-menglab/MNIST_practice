import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

#訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download = True)
#検証データ
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download = True)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#force cpu
#device = 'cpu' 
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
print(model)

# 損失関数　criterion：基準
# CrossEntropyLoss：交差エントロピー誤差関数
criterion = nn.CrossEntropyLoss()

# 最適化法の指定　optimizer：最適化
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, device='cpu'):

    train_loss = 0.0
    train_correct = 0
    num_train = 0

    # 学習モデルに変換
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        # batch数をカウント
        num_train += len(labels)

        images, labels = images.view(-1, 28*28).to(device), labels.to(device)

        # 勾配を初期化
        optimizer.zero_grad()

        # 推論(順伝播)
        outputs = model(images)

        # 損失の算出
        loss = criterion(outputs, labels)

        # 誤差逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
    
    # lossの平均値を取る
    train_loss = train_loss / num_train
    train_correct = float(train_correct) / num_train

    return train_loss, train_correct


def test_model(model, test_loader, criterion, optimizer, device='cpu'):

    test_loss = 0.0
    test_correct = 0
    num_test = 0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化
        for i, (images, labels) in enumerate(test_loader):
            num_test += len(labels)
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)

        # lossの平均値を取る
        test_loss = test_loss / num_test
        test_correct = float(test_correct) / num_test
    return test_loss, test_correct

def lerning(model, train_loader, test_loader, criterion, opimizer, num_epochs, device='cpu'):

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs+1, 1):

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device=device)
        test_loss, test_acc = test_model(model, test_loader, criterion, optimizer, device=device)
        
        print("epoch : {}, train_loss : {:.5f}, train_acc : {:.5f}, test_loss : {:.5f}, test_acc : {:.5f}" .format(epoch, train_loss, train_acc, test_loss, test_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
    
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list

num_epochs = 15
train_loss_list, train_acc_list, test_loss_list, test_acc_list = lerning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)

fig1 = plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()
fig1.savefig("rate.png")

fig2 = plt.figure()
plt.plot(range(len(train_acc_list)), train_acc_list, c='b', label='train acc')
plt.plot(range(len(test_acc_list)), test_acc_list, c='r', label='test acc')
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.grid()
plt.show()
fig2.savefig("acc.png")

def confusion_matrix(model,name,reshape):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    cm = torch.zeros(10, 10)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if reshape:
                inputs = inputs.view(inputs.shape[0],-1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    cm = cm.numpy()
    for i in range(10):
        cm[i] = cm[i]/np.sum(cm[i])
    cm = np.around(cm,3)

    fig3 = plt.figure(figsize=(10,7))
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sn.set(font_scale=1.3)
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 12}, cmap='Blues')
    plt.suptitle(name + ' Confusion Matrix', fontsize=16)
    plt.show()
    fig3.savefig(name + 'confusion_matrix.png')

confusion_matrix(model,'Pytorch Reimplemetation',reshape=True)

print("Brakepoint")
