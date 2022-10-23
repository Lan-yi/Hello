import os
import cv2
import time
import datetime
from diagnose_logging import Logger
from PIL import Image
import numpy as np

log = Logger('utils.py')
logger = log.getlog()

# class batchCap():
#     '''
#     从摄像头采集数据，默认采取100张，也可以按一下s键采一张
#     '''
#     def __init__(self):
#         self.save_path = './capframes/'
#         self.save_num = 100
#
#     def videoCap(self):
#         cap = cv2.VideoCapture(0)                    # 计算机自带的摄像头为0，外部设备为1
#         i = 0
#         while True:
#             ret, frame = cap.read()                  # ret:True/False,代表有没有读到图片  frame:当前截取一帧的图片
#             cv2.imshow("capture", frame)
#
#             # if (cv2.waitKey(1) & 0xFF) == ord('s'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
#             #     #转换成灰度图保存
#             #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
#             #     # gray = cv2.resize(gray, (224, 224))  # 图像大小为320*240
#             #     resize = cv2.resize(frame, (224, 224))  # 图像大小为320*240
#             #     cv2.imwrite('{}{}.jpg'.format(self.save_path, str(i)), resize)
#             #     i += 1
#             resize = cv2.resize(frame, (224, 224))  # 图像大小为320*240
#             cv2.imwrite('{}{}.jpg'.format(self.save_path, str(i)), resize)
#             time.sleep(0.1)  # 1s钟10张图片
#             i += 1
#             if i == self.save_num:
#                 break
#             if (cv2.waitKey(1) & 0xFF) == ord('q'):
#                 break
#
#         cap.release()
#         cv2.destroyAllWindows()


class batchRename():
    """
    批量重命名文件夹中的图片文件
    """
    def __init__(self):
        self.path = './capframes/bak/'         # 表示需要命名处理的主文件夹
        self.start_i = 1                       # 表示图片文件的命名是从1开始的

    def rename(self):
        listFileMain = os.listdir(self.path)  # 获取文件路径
        for listFile in listFileMain:
            listFileSub = os.listdir(self.path + listFile)
            for item in listFileSub:
                if item.endswith('.bmp'):     # 转换格式就可以调整为自己需要的格式即可
                    src = os.path.join(self.path + listFile, item)
                    dst = os.path.join(self.path + listFile, '' + str(self.start_i) + '.bmp')
                    # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                    # 这种情况下的命名格式为0000000.jpg形式，可以自定义格式
                    try:
                        os.rename(src, dst)
                        print('converting %s to %s ...' % (src, dst))
                        self.start_i += 1
                    except:
                        continue
        print('rename & converted %d jpgs' % (self.start_i))


class imgAug():
    def __init__(self, pathSrc, pathDst):
        self.pathSrc = pathSrc  # 图像完整路径
        self.pathDst = pathDst
        self.start_num = 500

        # 创建输出根目录
        try:
            if not os.path.exists(pathDst):
                os.mkdir(pathDst)
        except Exception as e:
            logger.error(e)
        logger.info('ImgPre: %s', pathSrc)

    def get_savename(self, operate):
        """
        :param export_path_base: 图像输出路径
        :param operate: 脸部区域名
        :return: 返回图像存储名
        """
        try:
            saveName = operate
            return saveName

        except Exception as e:
            logger.error('get_saveName ERROR')
            logger.error(e)

    def lightness(self, pathRead, pathSaveDir, light_list):
        """改变图像亮度.
        推荐值：
            0.87，1.07
        明亮程度
            darker < 1.0 <lighter
        """
        try:
            operate = 'lightness_'
            for light in light_list:
                with Image.open(pathRead) as image:         # 打开图像完整路径+图像
                    # 图像左右翻转
                    out = image.point(lambda p: p * light)
                    # 重命名
                    #savename = self.get_savename(operate)
                    # 图像存储
                    out.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1
            #日志
            logger.info("{}: current picture number is {}".format(operate, self.start_num))

        except Exception as e:
            logger.error('ERROR %s', operate)
            logger.error(e)

    def rotate(self,  pathRead, pathSaveDir, angle_list):
        """图像旋转15度、30度."""
        try:
            operate = 'rotate_'
            for angle in angle_list:
                with Image.open(pathRead) as image:  # 打开图像完整路径+图像
                    out = image.rotate(angle)
                    # 图像存储
                    out.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1
            logger.info("{}: current picture number is {}".format(operate, self.start_num))

        except Exception as e:
            logger.error('ERROR %s', operate)
            logger.error(e)

    def transpose(self, pathRead, pathSaveDir):
        """图像左右翻转操作."""
        try:
            operate = 'transpose'
            with Image.open(pathRead) as image:
                # 图像左右翻转
                out = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 图像存储
                out.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                self.start_num += 1
            logger.info("{}: current picture number is {}".format(operate, self.start_num))

        except Exception as e:
            logger.error('ERROR %s', operate)
            logger.error(e)

    def deform(self, pathRead, pathSaveDir):
        """图像拉伸."""
        try:
            operate = 'deform'
            with Image.open( pathRead) as image:
                w, h = image.size
                w, h = int(w), int(h)
                # 拉伸成宽为w的正方形
                out_ww = image.resize((int(w), int(w)))
                out_ww.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                self.start_num += 1
                # 拉伸成宽为h的正方形
                out_ww = image.resize((int(h), int(h)))
                out_ww.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                self.start_num += 1
            logger.info("{}: current picture number is {}".format(operate, self.start_num))

        except Exception as e:
            logger.error('ERROR %s', operate)
            logger.error(e)

    def crop(self, pathRead, pathSaveDir, scale_list):
        """提取四个角落和中心区域."""
        try:
            operate = 'crop'
            with Image.open(pathRead) as image:
                w, h = image.size
                # 切割后尺寸
                for scale in scale_list:
                    ww ,hh= int(w * scale), int(h * scale)
                    x = y = 0      # 图像起点，左上角坐标
                    # 切割左上角
                    x_lu ,y_lu = x, y
                    out_lu = image.crop((x_lu, y_lu, ww, hh))
                    out_lu.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1

                    # 切割左下角
                    x_ld ,y_ld= int(x), int(y + (h - hh))
                    out_ld = image.crop((x_ld, y_ld, ww, hh))
                    out_ld.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1

                    # 切割右上角
                    x_ru ,y_ru= int(x + (w - ww)), int(y)
                    out_ru = image.crop((x_ru, y_ru, w, hh))
                    out_ru.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1

                    # 切割右下角
                    x_rd ,y_rd= int(x + (w - ww)), int(y + (h - hh))
                    out_rd = image.crop((x_rd, y_rd, w, h))
                    out_rd.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1

                    # 切割中心
                    x_c ,y_c= int(x + (w - ww) / 2), int(y + (h - hh) / 2)
                    out_c = image.crop((x_c, y_c, ww, hh))
                    out_c.save(pathSaveDir + '/' + str(self.start_num) + '.jpg', quality=100)
                    self.start_num += 1
            logger.info("{}: current picture number is {}".format(operate, self.start_num))

        except Exception as e:
            logger.error('ERROR %s', operate)
            logger.error(e)

    def batchAug(self):
        listFile = os.listdir(self.pathSrc)            # 获取文件路径
        for i ,pic in enumerate(listFile):        # 获取图片个数
            pathSaveDir = self.pathDst + str(i+1)
            if not os.path.exists(pathSaveDir):   # 创建子文件夹
                os.mkdir(pathSaveDir)
            if pic.endswith('.jpg'):
                pathRead = self.pathSrc + pic             #图片全路径+名称

                # light_list = [0.67, 0.77, 0.87, 0.97, 1.07, 1.17, 1.27, 1.37]  # 亮度操作
                # light_list = [0.67]  # 亮度操作
                # self.lightness(pathRead, pathSaveDir, light_list)

                angle_list = [135,225]                          #角度旋转操作
                self.rotate(pathRead, pathSaveDir, angle_list)

                # self.transpose(pathRead, pathSaveDir)                          #左右翻转操作

                # self.deform(pathRead, pathSaveDir)                             #图像拉伸操作

                # scale_list = [0.85, 0.875, 0.9, 0.925]
                # scale_list = [0.85]
                # self.crop(pathRead, pathSaveDir, scale_list)                   #图像裁剪操作

            logger.info("The {}th picture {} is finished".format(i, pic))


def imgAug_batch():
    # 源地址和输出地址
    pathSrc = './capframes/face_src/'
    pathDst = './capframes/face_dst/'
    # 声明类对象
    imgPre = imgAug(pathSrc, pathDst)
    imgPre.batchAug()



if __name__ == '__main__':
    print('start...')
    start_time = datetime.datetime.now()
    #Cap = batchCap()         #摄像头采集图片
    #Cap.videoCap()
    # Rename = batchRename()   #图像批量重命名
    # Rename.rename()
    imgAug_batch()            #图像增强

    end_time = datetime.datetime.now()
    time_consume = (end_time - start_time).microseconds / 1000000

    logger.info('start_time: %s', start_time)
    logger.info('end_time: %s', end_time)
    logger.info('time_consume: %s(s)', time_consume)  # 0.280654(s)
