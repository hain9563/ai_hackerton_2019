import random
import os
import time
import sys
import argparse
import numpy
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader

import nsml
from nsml import DATASET_PATH, GPU_NUM


from dataprocessing import FundusDataset, FundusTestDataset
from dataprocessing2 import dataset_loader, image_preprocessing

## setting values of preprocessing parameters
RESIZE = 10.

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        # if init_type == 'normal':
        #     torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        # elif init_type == 'xavier':
        #     torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
        # elif init_type == 'kaiming':
        #     torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bind_model(model, optimizer, lr_sche, batch_size):
    def save(dir_name):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sche': lr_sche.state_dict()
        }
        os.makedirs(dir_name, exist_ok=True)
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('model saved!')

    def load(dir_name):
        state = torch.load(os.path.join(dir_name, 'model,pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        if 'lr_sche' in state and lr_sche:
            lr_sche.load_state_dict(state['lr_sche'])
        print('model loaded!')

    def infer(data, p_list, resize_factor=RESIZE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        images = []
        for i, d in enumerate(data):
            # test 데이터를 training 데이터와 같이 전처리 하기
            images.append(image_preprocessing(d, resize_factor, p_list[i]))

        model.eval()
        test_data_loader = DataLoader(FundusTestDataset(images), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            predResults = []
            for X in test_data_loader:
                pred = model(X)
                pred = pred.data.max(1)[1]  # 모델 예측 결과: 0-3
                for p in pred:
                    predResults.append(p.item())

        print('Prediction done!\n Saving the result...')
        return numpy.array(predResults)

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=10)  # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=1)  # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)  # DO NOT CHANGE num_classes, class 수는 항상 4

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # CUDA Device Setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed(777)

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes

    """ Model """

    learning_rate = 0.0001
    momentum = 0.9
    nesterov = True
    milestones = [50, 60]
    gamma = 0.1

    from models.EfficientNet import EfficientNet
    from models.Classifier import CI

    model = nn.Sequential(EfficientNet.from_name('efficientnet-b0').apply(weights_init_normal),
                          CI(1280 * 7 * 7, -1).apply(weights_init_normal)
                          )
    if torch.cuda.device_count() > 1:  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)  # 데이터 병렬 처리 설정
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(.5, .9))#torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov)
    lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    bind_model(model, optimizer, lr_sche, batch_size)

    if config.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train':  ### training mode일 때
        print('Training Start...')

        img_path = DATASET_PATH + '/train/'
        # images, labels, t_images, t_labels = dataset_loader(, resize_factor=RESIZE)
        #
        # data_loader = DataLoader(Data(images, labels, 64), batch_size=batch_size, shuffle=True)
        # test_data_loader = DataLoader(Data(t_images, t_labels, 64), batch_size=batch_size, shuffle=False)

        """ Training loop """

        t0 = time.time()
        RESIZE = 10.
        RESCALE = True


        for epoch in range(nb_epoch):
            t1 = time.time()
            loss, acc = 0, 0
            num_runs = 0
            model.train()
            train_sum = 0
            test_sum = 0
            for i in range(10):
                img, labels = dataset_loader(img_path, idx=i, resize_factor=RESIZE, rescale=RESCALE)
                print("### Model Fitting.. ###")
                print('epoch = {} / {}, idx {}'.format(epoch, nb_epoch, i))
                print('chaeck point = {}'.format(epoch))

                if i < 9:
                    data_loader = DataLoader(FundusDataset(img, labels), batch_size=64, shuffle=True)

                    train_sum += len(data_loader.dataset)
                    for i, (X, Y) in enumerate(data_loader):
                        X = X.to(device)
                        Y = Y.to(device)

                        prediction = model(X)  # Forward Propagation
                        optimizer.zero_grad()  # Gradient 초기화
                        cost = criterion(prediction, Y)  # Cost 계산
                        cost.backward()  # Back Propagation
                        optimizer.step()  # Update Parameters

                        pred_cls = prediction.data.max(1)[1]
                        acc += pred_cls.eq(Y.view(-1).cuda(0).data).cpu().sum()
                        loss += cost.item()
                        num_runs += 1
                        print('train, step {}, loss {}'.format(i, cost.item()))
                        # opt_params = optimizer.state_dict()['param_groups'][0]

                    nsml.report(
                        summary=True,
                        epoch=epoch,
                        epoch_total=nb_epoch,
                        train__loss=float(loss) / train_sum,
                        train__accuracy=float(acc) / train_sum
                    )

                    if (lr_sche):
                        lr_sche.step()
                    else:
                        pass
                    t2 = time.time()
                print('Training time for one epoch : %.1f' % ((t2 - t1)))

                """ Test Model """
                if i >= 9:
                    test_data_loader = DataLoader(FundusDataset(img, labels), batch_size=64, shuffle=False)
                    test_sum += len(test_data_loader.dataset)
                    model.eval()
                    loss, acc = 0, 0
                    num_runs = 0
                    with torch.no_grad():
                        for X_test, Y_test in test_data_loader:
                            X_test = X_test.to(device)
                            Y_test = Y_test.to(device)

                            prediction = model(X_test)
                            cost = criterion(prediction, Y_test)

                            pred_cls = prediction.data.max(1)[1]
                            acc += pred_cls.eq(Y_test.view(-1).cuda(0).data).cpu().sum()
                            loss += cost.item()
                            num_runs += 1
                            print("test step {}".format(num_runs))


                    nsml.report(
                        summary=True,
                        epoch=epoch,
                        epoch_total=nb_epoch,
                        test__loss=float(loss) / test_sum,
                        test__accuracy=float(acc) / test_sum
                    )
                nsml.save(epoch)

            print('Total training time : %.1f' % (time.time() - t0))