import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
# import track_test_notebook
import webbrowser
import os
import utils
import visualize
from visualize import display_images
import model as modellib
from config import Config
from model import log
import detector
import shutil
import seaborn as sns
import pandas as pd
import imageio
import norfair
from norfair import Detection, Paths, Tracker, Video
from typing import List

ROOT_DIR = os.path.abspath(".")
print(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models\logs")
print(MODEL_DIR)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models\mask_rcnn_taco_0100.h5")
print(COCO_MODEL_PATH)

import csv
import dataset

# Load class map - these tables map the original TACO classes to your desired class system
# and allow you to discard classes that you don't want to include.
class_map = {}
with open("./taco_config/map_10.csv") as csvfile:
    reader = csv.reader(csvfile)
    class_map = {row[0]: row[1] for row in reader}

TACO_DIR = "../data"
round = None  # Split number: If None, loads full dataset else if int > 0 selects split no
subset = "test"  # Used only when round !=None, Options: ('train','val','test') to select respective subset
dataset = dataset.Taco()
taco = dataset.load_taco(TACO_DIR, round, subset, class_map=class_map, return_taco=True)
# Must call before using the dataset
dataset.prepare()


class TacoTestConfig(Config):
    NAME = "taco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.75
    NUM_CLASSES = dataset.num_classes


config = TacoTestConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=TACO_DIR, config=config)
print(MODEL_DIR)
# model_path="./models/logs/tacoplus/mask_rcnn_taco_0006.h5"
# model_path="./models/logs/tacoplus/mask_rcnn_taco_0100.h5"
model_path = "./models/logs/tacoplus/mask_rcnn_taco_plus_0100.h5"

model.load_weights(weights_in_path=model_path, weights_out_path=model_path, by_name=True)

iou = []
dicescores = []
precisions = []
recalls = []
mdicescores = []
mprecisions = []
mrecalls = []
boxposition = []
cboxscores = []
outfile = open("statistics.txt", "w")
outfile.write("Scores for Detections \n")
# for x in range(len(dataset.image_ids)):
for x in range(10):

    save_dir = ("./saves/" + str(x) + 'ground' + ".png")
    outfile.write("Image ")
    outfile.write(str(x))
    outfile.write("\n")
    # image_id = np.random.choice(dataset.image_ids)
    image_id = x
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)
    height, width = image.shape[:2]
    # print(height,width)
    visualize.display_instances(image, save_dir, bbox, mask, class_ids, dataset.class_names,
                                figsize=(width / 77, height / 77))
    outfile.write("Actual number of obejcts:")
    outfile.write(str(len(bbox)))
    outfile.write('\n')

    results = model.detect([image], verbose=1)
    r = results[0]

    save_dir = ("./saves/" + str(x) + 'pred' + ".png")

    visualize.display_instances(image, save_dir, r['rois'], r['masks'], r['class_ids'], dataset.class_names,
                                r['scores'], figsize=(width / 77, height / 77))
    outfile.write("Number of objects detected:")
    outfile.write(str(len(r['rois'])))
    outfile.write('\n')
    # print(r['rois'])
    iou.append([])
    boxposition.append([])
    # visualize.display_differences(image, save_dir, bbox, class_ids, mask, r['rois'], r['class_ids'], r['scores'], r['masks'], dataset.class_names)
    for y in range(len(bbox)):
        u = 0
        # print(x)
        groundarea = (bbox[y][3] - bbox[y][1]) * (bbox[y][2] - bbox[y][0])
        # print(groundarea)
        detected = False
        while (len(r['rois'] > 0) and u < len(r['rois'])):
            if (detected == False):
                pbox = r['rois'][u]
                predarea = (r['rois'][u][3] - r['rois'][u][1]) * (r['rois'][u][2] - r['rois'][u][0])
                intersection = []
                # print()
                # print("pbox ",pbox)
                # print("bbox ",bbox[y])
                if (pbox[0] <= bbox[y][2] and pbox[0] >= bbox[y][0]):
                    intersection.append(pbox[0])
                elif (bbox[y][0] <= pbox[2] and bbox[y][0] >= pbox[0]):
                    intersection.append(bbox[y][0])

                if (pbox[1] <= bbox[y][3] and pbox[1] >= bbox[y][1]):
                    intersection.append(pbox[1])
                elif (bbox[y][1] <= pbox[3] and bbox[y][1] >= pbox[1]):
                    intersection.append(bbox[y][1])

                if (pbox[2] <= bbox[y][2] and pbox[2] >= bbox[y][0]):
                    intersection.append(pbox[2])
                elif (bbox[y][2] <= pbox[2] and bbox[y][2] >= pbox[0]):
                    intersection.append(bbox[y][2])

                if (pbox[3] <= bbox[y][3] and pbox[3] >= bbox[y][1]):
                    intersection.append(pbox[3])
                elif (bbox[y][3] <= pbox[3] and bbox[y][3] >= pbox[1]):
                    intersection.append(bbox[y][3])

                # print("intersection",intersection)
                if (len(intersection) == 4):
                    inter = (intersection[3] - intersection[1]) * (intersection[2] - intersection[0])
                    union = predarea + groundarea - inter
                    IoU = inter / union
                    if IoU > 0.5:
                        iou[x].append(IoU)
                        boxposition[x].append(u)
                        detected = True

                    print(iou[x])
                    print()

            u += 1
        if detected == False:
            boxposition[x].append(-1)
    print(u)
    if (len(r['rois'] > 0)):
        precision = (len(iou[x])) / len(r['rois'])
    else:
        precision = 0
    recall = (len(iou[x])) / len(bbox)
    dice = (2 * len(iou[x])) / (len(bbox) + len(r['rois']))

    outfile.write('bbox\n')
    outfile.write("Precision: ")
    outfile.write(str(precision))
    outfile.write("\n")

    outfile.write("Recall: ")
    outfile.write(str(recall))
    outfile.write("\n")

    outfile.write("Dice score: ")
    outfile.write(str(dice))
    outfile.write("\n\n")

    precisions.append(precision)
    recalls.append(recall)
    dicescores.append(dice)
    miou = []

    if (len(r['rois']) > 0):
        ov = utils.compute_overlaps_masks(r['masks'], mask)

        # print(overlaps)
        # print(gt_match)
        # print(pred_match)
        counter = 0
        for item in boxposition[x]:
            if (item >= 0):
                miou.append(ov[item][counter])
            counter += 1

        # print(len(bbox))
        mprecision = (len(miou)) / len(r['rois'])
        mrecall = (len(miou)) / len(bbox)
        mdice = (2 * len(miou)) / (len(bbox) + len(r['rois']))
    else:
        mprecision = 0
        mrecall = 0
        mdice = 0

    # print(ov)
    mdicescores.append(mdice)
    mprecisions.append(mprecision)
    mrecalls.append(mrecall)
    outfile.write('Mask\n')
    outfile.write("Precision: ")
    outfile.write(str(mprecision))
    outfile.write("\n")

    outfile.write("Recall: ")
    outfile.write(str(mrecall))
    outfile.write("\n")

    outfile.write("Dice score: ")
    outfile.write(str(mdice))
    outfile.write("\n\n")

    print('bbox:')
    print('precision ', precisions)
    print('recall ', recalls)
    print('dice scores ', dicescores)
    print()

    print('masks:')
    print('precision ', mprecisions)
    print('recall ', mrecalls)
    print('dice scores ', mdicescores)
    print()

    print('bbox:')
    print('precision', sum(precisions) / len(precisions))
    print('recall', sum(recalls) / len(recalls))
    print('dice score', sum(dicescores) / len(dicescores))
    print()

    print("mask:")
    print('precision', sum(mprecisions) / len(mprecisions))
    print('recall', sum(mrecalls) / len(mrecalls))
    print('dice score', sum(mdicescores) / len(mdicescores))
    print()

    # Including Classification
    grounds, preds, cmAp, cprec, crec, boxlap, cov = utils.compute_ap1(bbox, class_ids, mask, r['rois'], r['class_ids'],
                                                                       r['scores'], r['masks'],
                                                                       iou_threshold=0.5)
    print("masks")
    print('cmap', cmAp)
    print('precision', sum(cprec) / len(cprec))
    print('recall', sum(crec) / len(crec))

    print('boxes')
    print(preds)
    print(grounds)
    print(boxlap)
    cboxscores.append([])
    for item in grounds:
        for content in preds:
            if (item >= 0 and content >= 0):
                cboxscores[x].append(boxlap[int(item)][int(content)])
    print(cboxscores)

    if (len(r['rois'] > 0)):
        cprecision = (len(cboxscores[x])) / len(r['rois'])
    else:
        precision = 0
    crecall = (len(cboxscores[x])) / len(bbox)
    cdice = (2 * len(cboxscores[x])) / (len(bbox) + len(r['rois']))

    print('precision', cprecision)
    print('recall', crecall)
    print('dice score', cdice)

outfile.write("Average Precision: ")
outfile.write(str(sum(precisions) / len(precisions)))
outfile.write("\n")

outfile.write("Average Recall: ")
outfile.write(str(sum(recalls) / len(recalls)))
outfile.write("\n")

outfile.write("Average Dice Score: ")
outfile.write(str(sum(dicescores) / len(dicescores)))
