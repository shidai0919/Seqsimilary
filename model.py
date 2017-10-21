#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
transform=transforms.Compose([transforms.ToTensor(),
                             ])
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, train=True,download=False, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.train = train
    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
fr = open('./d+d-array.pkl')    #open的参数是pkl文件的路径
data_tensor = pickle.load(fr)
# fa = open('./testarray.pkl')    #open的参数是pkl文件的路径
# datatest = pickle.load(fa)
# fb = open('./testtarget.pkl')
# targettest = pickle.load(fb)
# 将数据封装成Dataset
trainset = TensorDataset(data_tensor, train=True, download=False, transform=transform)

#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=False, num_workers=2)

#测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
# testset = TensorDataset(datatest, targettest, train=False, download=False, transform=transform)
#
# #将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
# testloader = torch.utils.data.DataLoader(testset, batch_size=8,
#                                           shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(6*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6*5*5)
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

def lmax(match):
    max(0,1-match[0]+match[1])#叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9

for epoch in range(2): # 遍历数据集两次

    running_loss = 0.0
    #enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs= data   #data的结构是：[4x3x32x32的张量,长度4的张量]

        # wrap them in Variable
        inputs = Variable(inputs)  #把input数据从tensor转为variable

        # zero the parameter gradients
        optimizer.zero_grad() #将参数的grad值初始化为0

        # forward + backward + optimize
        outputs = net(inputs)
        if i%2==0:
            match=[]
            match.append(outputs)
        else:
            match.append(outputs)
            loss = lmax(match) #将output和labels使用叉熵计算损失
            loss.backward() #反向传播
            optimizer.step() #用SGD更新参数

        # 每2000批数据打印一次平均loss值
            running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
            if i % 20 == 19: # 每20批打印一次
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 20))
                running_loss = 0.0

print('Finished Training')

# correct = 0
# total = 0
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     #print outputs.data
#     _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
#     total += labels.size(0)
#     correct += (predicted == labels).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
