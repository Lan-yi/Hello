import os
import json
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import natsort
from model import resnet34

folder = '../data_set/medicaldata_uint16_png_3_2_4/'
dir =  'train/PCR/'
folder = folder + dir

files = os.listdir(folder)
files = natsort.natsorted(files)


def medical_predict(img_path,imgname):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(448),
         transforms.ToTensor()
         ])
    # data_transform = transforms.Compose(
    #     [transforms.Resize(256),
    #      transforms.CenterCrop(224),
    #      transforms.ToTensor(),
    #      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #      ])

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    #imgarray = np.array(img)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        #rint(output)   ###

        predict = torch.softmax(output, dim=0)
        #print(predict)
        predict_cla = torch.argmax(predict).numpy()
        #print(predict_cla)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    cla = class_indict[str(predict_cla)]
    print(imgname + ' : ' + print_res)
    return cla
    #plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    #plt.show()


nPCR_num = 0
PCR_num = 0
i = 0
for i in range(0,len(files)):
    img_path = folder + files[i]
    if(str(medical_predict(img_path, files[i]))) == 'nPCR':
        nPCR_num = nPCR_num + 1
    else:PCR_num = PCR_num + 1

    # if i > len(files):
    #     break
print('nPCR_num:' , nPCR_num)
print('PCR_num:' , PCR_num)


# if __name__ == '__main__':
#     main()