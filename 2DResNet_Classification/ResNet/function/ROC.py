import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score, accuracy_score

# y1=[0,1,0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]

with open("roc.txt", "r", encoding="utf-8") as f:
    y_true, y_sore, y_pre = [], [], []
    f = f.readlines()
    data = [i.split("\n")[0].split(" ") for i in f]
    print("# 第一列是真实值 第二列是分数 第三列是预测值", data)
    for line in data:
        y_true.append(int(line[0]))
        y_sore.append(float(line[1]))
        y_pre.append(int(line[2]))
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_pre)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(linestyle='-.')
plt.grid(True)
plt.show()
print(roc_auc)



Loss_list =[]#存储每次epoch损失值
def draw_loss(Loss_list,epoch):
#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    plt.cla()
    x1 = range(1, epoch+1)
    print(x1)
    y1 = Loss_list
    plt.title('Train loss vs.epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(".Train_loss.png")
    plt.show()



