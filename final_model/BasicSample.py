import random
import os
from model import getEggNet

from PIL import Image
import xlrd  # pip install xlrd
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#############################################################################
#   Class Definition

#****************************************************************************
#   Training 및 Test Dataset 생성을 위해
#   이미지 파일의 경로와 레이블의 list를 반환하는 Class
#****************************************************************************
class messidor2():
    def __init__(self, dataset_path, label_path, test_ratio=.1):
        self.label_p = label_path
        self.data_p = dataset_path
        self.all_imagepaths, self.all_labels = list(), list()
        self.ratio = test_ratio
        self._get_img()

    def _get_img(self):
        xls = xlrd.open_workbook(self.label_p)
        sheet = xls.sheet_by_index(0)
        sheet.row_values(0)
        lines = [line for line in [sheet.row_values(i) for i in range(1, sheet.nrows)]]

        random.seed(1234)
        random.shuffle(lines)

        for line in lines:
            self.all_imagepaths.append(os.path.join(self.data_p, line[0]))
            self.all_labels.append(int(line[2]))

        self.total_num = len(self.all_labels)
        self.test_size = int(self.total_num * self.ratio)
        # Make same testing set
        self.train_size = self.total_num - self.test_size
        

    def get_train_data(self):
        return self.all_imagepaths[self.test_size:], self.all_labels[self.test_size:]

    def get_test_data(self):
        return self.all_imagepaths[:self.test_size], self.all_labels[:self.test_size]


#****************************************************************************
#   Dataset 클래스. DataLoader에서 Data를 불러오도록 하기 위해 
#   __len__(), __getitem__() 메서드를 구현해야 함
#****************************************************************************
class dataset(Dataset):
    def __init__(self, img_list, label_list):
        resFactor = 4.

        self.img_list = []
        self.label_list = []
        self.size = len(label_list)

        transform = transforms.Compose([transforms.CenterCrop((int(960 * (3/4)), int(960 * (3/4)))),
                                    transforms.Resize((256, 256)),
                                    transforms.Grayscale(1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [1])])
        for index in range(self.size):
            img = Image.open(img_list[index])
            img = transform(img)
            lable = torch.tensor(label_list[index])
            self.img_list.append(img)
            self.label_list.append(lable)

    def __len__(self):
        return self.size

    #****************************************************************************
    #  이미지를 해당 경로에서 불러와 Crop한 뒤,
    #  이미지와 레이블을 Tensor로 변환한 후 반환
    #  Image Size : [3, 1487, 1487]
    #****************************************************************************
    def __getitem__(self, index):
        return (self.img_list[index], self.label_list[index])


#   Class Definition End
##################################################################################


#************************************
#   변수 설정
batch_size = 5
learning_rate = 0.0001
training_epochs = 10

momentum = 0.9
nesterov = True
milestones=[5,7]
gamma = 0.7
#************************************

#************************************
#   Dataset 및 Data Loader 생성
data = messidor2('./Base/', './Base/Annotation_Base.xls')
train_dataset = data.get_train_data()
test_dataset = data.get_test_data()

train_loader = DataLoader(dataset(train_dataset[0], train_dataset[1]), batch_size=batch_size)
test_loader = DataLoader(dataset(test_dataset[0], test_dataset[1]), batch_size=batch_size)
print('load finish')
#************************************


#************************************
#   CUDA (GPU) Device 설정 및 모델 생성
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

model = getEggNet(4)
if torch.cuda.device_count() > 1:       # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)      # 데이터 병렬 처리 설정
model = model.to(device)    # 해당 Tensor에 대한 연산을 GPU 에서 수행하도록 설정.
                            # Data(X), Label(Y)에 대해서도 이와 같은 코드를 추가해야 함
#************************************


#************************************
#   Loss Function 및 Weight Update 방식 지정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov)
lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
#************************************


#************************************
#   학습 수행
for epoch in range(training_epochs):
    i = 0
    loss, acc = 0, 0
    avg_cost = 0
    train_sum = len(train_loader.dataset)
    model.train()
    for X, Y in train_loader:
        X = X.to(device)    # GPU 연산 지정
        Y = Y.to(device)

        prediction = model(X)               # Forward Propagation
        optimizer.zero_grad()               # Gradient 초기화
        cost = criterion(prediction, Y)     # Cost 계산
        cost.backward()                     # Back Propagation
        optimizer.step()                    # Update Parameters
        
        pred_cls = prediction.data.max(1)[1]
        acc += pred_cls.eq(Y.view(-1).cuda(0).data).cpu().sum()
        loss += cost.item()

        i = i+1
        print(' batch : {}, cost : {}'.format(i, cost))

    if(lr_sche):
        lr_sche.step()

    print('epoch : {}, cost : {}, acc : {}'.format(epoch, float(loss)/train_sum, float(acc)/train_sum))

    #************************************
    #   테스트
    model.eval()
    with torch.no_grad():   # Gradient를 저장하지 않음을 의미 (테스트 시 Weight를 Update하지 않기 때문)
        acc = 0
        for X, Y in test_loader:
            X_test = X.to(device)
            Y_test = Y.to(device)
    
            prediction = model(X_test)

            pred_cls = prediction.data.max(1)[1]
            acc += pred_cls.eq(Y_test.view(-1).cuda(0).data).cpu().sum()
        print('accuracy : ', float(acc)/len(test_loader.dataset))


#************************************
print('Finished!')
#************************************


