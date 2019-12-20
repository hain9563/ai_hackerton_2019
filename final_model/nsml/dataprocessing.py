import cv2
from PIL import Image
import time
import os
import torch
import operator
import random
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def clahe_gridsize(brg, denoise=False, verbose=False, cliplimit=None, gridsize=50):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        #bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr

def RGBtoRG(cv_original):
    cv_rg = cv_original
    b, g, r = cv2.split(cv_rg)
    zeros = np.zeros((cv_rg.shape[0], cv_rg.shape[1]), dtype="uint8")
    cv_rg = cv2.merge([b, g, zeros])
    return cv_rg

class FundusDataset(Dataset):
    def __init__(self, data_list):
        self.size = len(data_list)
        self.data_list = []
        for index in range(self.size):
            im = cv2.imread(data_list[index][0])
            self.data_list.append((image_preprocessing(im), torch.tensor(data_list[index][1])))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data_list[index]


# No Lable
class FundusTestDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.size = len(img_list)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = self.img_list[index]
        return img


# ****************************************************
#   im : PIL Image
#   return : Tensor
def image_preprocessing(d):
    #######
    h, w, c = 3900, 3072, 3
    nh, nw = 256, 256
    ########

    # 오른쪽 눈으로 통일
    # if(p is not None):
    #    im = hor_filp(im, p)

    im = clahe_gridsize(d, gridsize=256)
    im = Image.fromarray(im)


    transform = transforms.Compose([transforms.CenterCrop((2240, 2240)),
                                    transforms.Resize((nh, nw)),
                                    transforms.ToTensor()])
    
    res = transform(im)

    return res


def Label2Class(label):
    if label == 'AMD':
        return 1
    elif label == 'RVO':
        return 2
    elif label == 'DMR':
        return 3
    else:
        return 0  # Normal


class ImagePathSet:
    def __init__(self, img_path, test_rate):
        self.train = []
        self.test = []

        image_labels = [[], [], [], []]

        p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if
                  all(s in f for s in ['.jpg'])]
        p_list.sort()
        for i, p in enumerate(p_list):
            l = Label2Class(p.split('/')[-2])
            image_labels[l].append(p)

        # Test 데이터셋 추출
        for _ in range(4):
            iter = len(image_labels[_])
            for __ in range(int(iter*test_rate)):
                i = random.randint(0, len(image_labels[_]) - 1)
                self.test.append([image_labels[_][i], _, False])
                del image_labels[_][i]

        # Sync Sampling
        for p in image_labels[2]:
            self.train.append([p,2,True])
        iter = len(image_labels[3])
        for p in range(iter//3):
            i = random.randint(0, len(image_labels[3])-1)
            del image_labels[3][i]

        # Train 데이터셋 생성 및 preprocessing
        for _ in range(4):
            for p in image_labels[_]:
                self.train.append([p, _, False])
        random.shuffle(self.train)

    def get_train(self, idx, total):
        return self.train[int(len(self.train) * (idx / total)):int(len(self.train) * ((idx + 1) / total))]

    def get_test(self):
        return self.test
