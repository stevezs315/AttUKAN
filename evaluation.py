import matplotlib
import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import argparse
parser = argparse.ArgumentParser(description="nasopharyngeal training")
parser.add_argument('--mode', default='gpu', type=str, metavar='train on gpu or cpu',
                    help='train on gpu or cpu(default gpu)')
parser.add_argument('--gpu', default=0, type=int, help='gpu number')
args = parser.parse_args()
mode = args.mode
gpuid = args.gpu
if mode == 'gpu':
    torch.cuda.set_device(gpuid)
    torch.cuda.manual_seed(0)
def calculate_connectivity(S, SG):
    """
    Calculate the connectivity factor C(S, SG) between the segmentation S and the reference segmentation SG.

    Parameters:
    S (ndarray): The segmentation to be evaluated.
    SG (ndarray): The reference segmentation.

    Returns:
    float: The connectivity factor C(S, SG).
    """
    # Calculate the number of connected components in SG and S
    num_components_SG, _ = label(SG)
    num_components_S, _ = label(S)

    # Extract the number of connected components (num_components_SG and num_components_S)
    num_components_SG = np.max(num_components_SG)
    num_components_S = np.max(num_components_S)

    # Calculate the cardinality of SG
    cardinality_SG = np.sum(SG)

    # Calculate the connectivity factor
    connectivity = 1 - min(1, abs(num_components_SG - num_components_S) / cardinality_SG)

    return connectivity

def calculate_area(segmentation, reference, dilation_radius=2):
    seg_dilated = cv2.dilate(segmentation.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
    2 * dilation_radius + 1, 2 * dilation_radius + 1)))
    ref_dilated = cv2.dilate(reference.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
    2 * dilation_radius + 1, 2 * dilation_radius + 1)))

    intersection_left = np.logical_and(segmentation, ref_dilated)
    intersection_right = np.logical_and(reference, seg_dilated)
    inter = np.logical_or(intersection_left, intersection_right)
    union = np.logical_or(segmentation, reference)
    A = np.sum(inter) / np.sum(union)
    return A

def calculate_length(segmentation, reference, dilation_radius=2):
    seg_skeleton = skeletonize(segmentation)
    ref_skeleton = skeletonize(reference)

    seg_dilated = cv2.dilate(segmentation.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1)))
    ref_dilated = cv2.dilate(reference.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1)))

    intersection_left = np.logical_and(seg_skeleton, ref_dilated)
    intersection_right = np.logical_and(ref_skeleton, seg_dilated)
    inter = np.logical_or(intersection_left,intersection_right)
    union = np.logical_or(seg_skeleton,ref_skeleton)
    L = np.sum(inter) / np.sum(union)
    return L

matplotlib.use("Agg")
from matplotlib import pyplot as plt
# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

from medpy.metric.binary import dc,jc,hd,hd95
from scipy.ndimage import label
import sys
from tqdm import tqdm

sys.path.insert(0, './utils/')
from utils.help_functions import *
from utils.extract_patches import pred_only_FOV, img_pred_only_FOV
# ========= CONFIG FILE TO READ FROM =======
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
# ===========================================
# model name
test_path = config.get('data attributes', 'dataset')
lr = float(config.get('training settings','lr'))
path_data = './'+test_path+config.get('data paths', 'path_local')
algorithm_config = config.get('experiment name', 'name')
name_experiment_list = [algorithm_config]
algorithms = config.get('experiment name', 'name')
test_border_masks = path_data+ test_path + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(test_border_masks)
def to_cuda(t, mode='gpu'):
    if mode == 'gpu':
        return t.cuda()
    return t

index = 0
for name_experiment in name_experiment_list:
    path_experiment = './log/experiments/' + name_experiment + '/' + test_path + '/'

    # ========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zion
    ## back to original dimensionsero!!
    # kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualizat
    if test_path == '':
        file = h5py.File(path_experiment + '0:15/' + dataset + '_predict_results.h5', 'r')
        gtruth_masks = file['y_gt'][:]
        pred_imgs = file['y_pred'][:]
        orig_imgs = file['x_origin'][:]
        file.close()
        file = h5py.File(path_experiment + '15:30/' + dataset + '_predict_results.h5', 'r')
        gtruth_masks = np.concatenate([gtruth_masks, file['y_gt'][:]], axis=0)
        pred_imgs = np.concatenate([pred_imgs, file['y_pred'][:]], axis=0)
        pred_imgs = to_cuda(torch.tensor(pred_imgs),'gpu')
        file.close()
        gtruth_masks = np.where(gtruth_masks > 0, 1, 0)
        gtruth_masks = to_cuda(torch.tensor(gtruth_masks),'gpu')
    else:
        print(path_experiment + 'predict_results.h5')
        file = h5py.File(path_experiment +  'predict_results.h5', 'r')
        gtruth_masks = file['y_gt'][:]
        pred_imgs = file['y_pred'][:]
        orig_imgs = file['x_origin'][:]
        file.close()

    # ====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    print('\n', name_experiment)
    print(path_experiment)
    num_predimgs = pred_imgs.shape[0]
    AUC_ROC_all=0
    AUC_prec_rec_all=0
    accuracy_all = 0
    specificity_all = 0
    sensitivity_all = 0
    precision_all = 0
    jaccard_index_all =0
    hd95_score_all = 0
    F1_score_all = 0
    connectivity_all = 0
    area_all = 0
    length_all = 0
    CAL_all = 0
    for i in tqdm(range(num_predimgs)):
        # predictions only inside the FOV
        y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks,
                                        insideFOV=True,ord=i)
        # returns data only inside the FOV
        if not np.array_equal(y_true, y_true.astype(bool)):
            y_true = np.where(y_true > 0.5, 1, 0)

        if np.any(y_scores < 0) or np.any(y_scores > 1):
            raise ValueError("y_scores not in [0, 1]")

        if np.max(y_true) > 1:
            y_true = y_true // np.max(y_true)

        # Area under the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        AUC_ROC_all += roc_auc_score(y_true, y_scores)
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration

        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        AUC_prec_rec_all += np.trapz(precision, recall)

        # Confusion matrix
        threshold_confusion = 0.5

        y_pred = np.empty((y_scores.shape[0]))

        for j in range(y_scores.shape[0]):
            if y_scores[j] >= threshold_confusion:
                y_pred[j] = 1
            else:
                y_pred[j] = 0

        confusion = confusion_matrix(y_true, y_pred)

        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        accuracy_all += accuracy

        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        specificity_all += specificity

        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        sensitivity_all +=sensitivity

        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        precision_all +=precision

        y_scores, y_true = img_pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks,
                                             insideFOV=True, ord=i)
        y_pred = np.where(y_scores > threshold_confusion, 1, 0)

        # Jaccard similarity index
        jaccard_index = jc(y_pred, y_true)
        jaccard_index_all += jaccard_index

        # hd95
        hd95_score = hd95(y_pred,y_true)
        hd95_score_all += hd95_score

        # F1 score
        F1_score = dc(y_pred, y_true)
        F1_score_all += F1_score

        #connectivity
        connectivity = calculate_connectivity(y_pred,y_true)
        connectivity_all += connectivity

        area = calculate_area(y_pred,y_true)
        area_all += area

        length = calculate_length(y_pred,y_true)
        length_all += length

        CAL = connectivity * area * length
        CAL_all += CAL

    AUC_ROC_all = AUC_ROC_all / num_predimgs
    AUC_prec_rec_all = AUC_prec_rec_all / num_predimgs
    accuracy_all = accuracy_all / num_predimgs
    specificity_all = specificity_all / num_predimgs
    sensitivity_all = sensitivity_all / num_predimgs
    precision_all = precision_all / num_predimgs
    jaccard_index_all = jaccard_index_all / num_predimgs
    hd95_score_all = hd95_score_all / num_predimgs
    F1_score_all = F1_score_all / num_predimgs
    connectivity_all = connectivity_all / num_predimgs
    area_all = area_all / num_predimgs
    length_all = length_all / num_predimgs
    CAL_all = CAL_all / num_predimgs


    print("Area under the ROC curve: " + str(AUC_ROC_all))
    print("Global Accuracy: " + str(accuracy_all))
    print("Sensitivity: " + str(sensitivity_all))
    print("F1 score (F-measure): " + str(F1_score_all))
    print("Jaccard similarity score: " + str(jaccard_index_all))
    print("hd95 score: " + str(hd95_score_all))
    print('C (connectivity):' + str(connectivity_all))
    print('A (area):' + str(area_all))
    print('L (length):' + str(length_all))
    print('CAL:' + str(CAL_all))

    # Save the results
    file_perf = open(path_experiment + 'performances_new.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC_all)
                    + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec_all)
                    + "\nJaccard similarity score: " + str(jaccard_index_all)
                    + "\nF1 score (F-measure): " + str(F1_score_all)
                    #+ "\n\nConfusion matrix:"
                    #+ str(confusion)
                    + "\nACCURACY: " + str(accuracy_all)
                    + "\nSENSITIVITY: " + str(sensitivity_all)
                    + "\nSPECIFICITY: " + str(specificity_all)
                    + "\nPRECISION: " + str(precision_all)
                    + "\nhd95: " + str(hd95_score_all)
                    + "\nconnectivity: " + str(connectivity_all)
                    + "\narea: " + str(area_all)
                    + "\nlength: " + str(length_all)
                    + "\nCAL: " + str(CAL_all)
                    )
    file_perf.close()
