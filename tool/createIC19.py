# covert icdar2019 to lmdb
import os
import cv2 as cv
import numpy as np
import math
from PIL import Image
import lmdb

debug = False

ignoreStr = '***'
def covertOneGt(gtPath):
    '''
    covert gt file to list
    :param gtPath:
    :return: labelList
    '''
    labelList = []

    with open(gtPath, 'r') as f:
        gtLines = f.readlines()
        bboxes = []
        for line in gtLines:
            parts = line.strip().split(',')
            box = [int(parts[i]) for i in range(8)]
            if len(parts) > 9:
                word = ','.join(parts[i] for i in range(8, len(parts)))
            else:
                word = parts[8]
            if ignoreStr != word:
                bboxes.append(box)
                labelList.append(word)

    return bboxes, labelList

def covertWordImg(img, bboxes):
    '''
    将一整张图片转为多个字符图片，通过仿射变换转为水平方向
    :param img:
    :param bboxes:
    :return:
    '''
    max_width = 0
    wordList = []
    for box in bboxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box

        rotated_rect = cv.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        # print(rotated_rect)
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]
        width = img.shape[1]
        height = img.shape[0]
        if box_w <= box_h:
            box_w, box_h = box_h, box_w
        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 32)

        width_box = math.ceil(32 * box_w / box_h)
        width_box = min(width_box, width)  # not to exceed feature map's width
        max_width = width_box if width_box > max_width else max_width

        mapped_x2, mapped_y2 = (width_box, 0)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([
            (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
        ])
        affine_matrix = cv.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))

        dst = cv.warpAffine(img, affine_matrix, (img.shape[0], img.shape[1]))
        wordRegion = dst[0:32, 0:width_box, :]
        wordList.append(covertToWebImgType(wordRegion))
    return wordList

def covertOneImg(path, imgName, imgSuffix):
    imgPath = os.path.join(path, imgName+'.'+imgSuffix)
    gtPath = os.path.join(path, imgName+'.txt')

    img = cv.imread(imgPath, 1)
    bboxes, labelList = covertOneGt(gtPath)
    wordList = covertWordImg(img, bboxes)

    return labelList, wordList



def covertOpencvImg2PIL(img):
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

def covertPILImg2Opencv(img):
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def covertToWebImgType(img, imgType = '.jpg'):
    img_encode = cv.imencode(imgType, img)[1]
    data_encode = np.array(img_encode)
    frame_encode = data_encode.tostring()
    return frame_encode

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv.imdecode(imageBuf, cv.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def covertIC19Dataset(path, outputPath, lexiconList=None, checkValid=True):


    imgList = []
    labelList = []
    suffix = ['jpg','png','jpeg', 'JPG']
    if not os.path.exists(path):
        raise RuntimeError("IC19 path is not exist")
    allFile = os.listdir(path)
    # prepare data
    for fileName in allFile:
        if debug:
            print(fileName)
        parts = fileName.strip().split('.')
        if parts[1] in suffix:
            singleLabelList, singleWordList = covertOneImg(path, parts[0], parts[1])
            labelList.extend(singleLabelList)
            imgList.extend(singleWordList)

    assert(len(imgList) == len(labelList))
    nSamples = len(imgList)


    # lmdb operation
    # env = lmdb.open(outputPath, map_size=1099511627776)/
    env = lmdb.open(outputPath, map_size=10737418240)
    cache = {}
    cnt = 1


    for i in range(nSamples):
        imageBin = imgList[i]
        label = labelList[i]

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % i)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)




if __name__ == '__main__':
    ic19TrainPath = 'D:/dataset/icdar2019/0325updated.task1train(626p)/'
    # img = cv.imread(ic19TrainPath+'X00016469612.jpg',1)
    # checkImageIsValid(covertToWebImgType(img))
    outputPath = './'
    covertIC19Dataset(ic19TrainPath, outputPath)
    # gtPath = os.path.join(ic19TrainPath, 'X00016469612.txt')
    # bboxes, labelList = covertOneGt(gtPath)
    # print(bboxes)
    # print(labelList)


    # covertIC19Dataset(ic19TrainPath)
    # box = np.array([177,247,544,247,544,292,177,292], np.int32)
    # x1, y1, x2, y2, x3, y3, x4, y4 = box
    #
    # box = box.reshape((-1,1,2))
    # img = cv.imread(ic19TrainPath+'X51007103599.jpg',1)
    # print(img.shape)
    # max_width = 0
    #
    # rotated_rect = cv.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
    # print(rotated_rect)
    # box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]
    # width = img.shape[1]
    # height = img.shape[0]
    # if box_w <= box_h:
    #     box_w, box_h = box_h, box_w
    # mapped_x1, mapped_y1 = (0, 0)
    # mapped_x4, mapped_y4 = (0, 32)
    #
    # width_box = math.ceil(32 * box_w / box_h)
    # width_box = min(width_box, width)  # not to exceed feature map's width
    # max_width = width_box if width_box > max_width else max_width
    #
    # mapped_x2, mapped_y2 = (width_box, 0)
    #
    # src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
    # dst_pts = np.float32([
    #     (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
    # ])
    # affine_matrix = cv.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    # print(affine_matrix)

    # point = cv.boxPoints(rotated_rect)
    # point = box.reshape((-1,1,2))
    # point = point.astype(np.int32)

    # cv.polylines(img, [box], True, (0,255,0))
    # cv.namedWindow('test', cv.WINDOW_NORMAL)
    # cv.imshow('test',img)
    # cv.waitKey()