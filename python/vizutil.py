"""
Created on Feb 28, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import itertools
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics
import tabulate

import config
import metadata


def plot_segmentation(input_labels_list, endframe):
    plt_idx = 0
    for input_labels in input_labels_list:
        seg_image = np.empty((10, endframe))

        for frame in range(endframe):
            seg_image[:, frame] = input_labels[frame]

        plt_idx += 1
        ax = plt.subplot(len(input_labels_list), 1, plt_idx)
        plt.imshow(seg_image)
    plt.show()


def visualize_tpg_labeling(gt_subactivity, gt_affordance, tpg, obj_num, end_frame):
    # Visualization of segmentation and labeling results for subactivity and affordance
    start_frame = tpg.terminals[0].start_frame
    end_frame = np.min([gt_subactivity.shape[0], tpg.terminals[-1].end_frame-start_frame, end_frame])
    # Get labels for every frame
    subactivity_lables = np.empty(end_frame, dtype=int)
    affordance_labels = np.empty((obj_num, end_frame), dtype=int)
    for spg in tpg.terminals:
        # Note: a spg spans [spg.start_frame, spg.end_frame], hence need to +1 in range()
        for frame in range(spg.start_frame, spg.end_frame+1):
            # print frame, spg.subactivity, metadata.subactivities[spg.subactivity]
            if frame >= end_frame + start_frame:
                break
            subactivity_lables[frame-start_frame] = spg.subactivity
            affordance_labels[:, frame-start_frame] = spg.affordance

    # Add labels to the plot list
    plot_labels = [gt_subactivity[:end_frame], subactivity_lables, (gt_subactivity[:end_frame]-subactivity_lables) == 0]
    for o in range(obj_num):
        plot_labels.append(gt_affordance[o, :end_frame])
        plot_labels.append(affordance_labels[o, :])
        plot_labels.append((gt_affordance[o, :end_frame]-affordance_labels[o, :]) == 0)
    plot_segmentation(plot_labels, end_frame)


def plot_confusion_matrix(cm, classes, filename=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    ax = plt.gca()
    ax.tick_params(axis=u'both', which=u'both', length=0)
    # matplotlib.rcParams.update({'font.size': 15})
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]), verticalalignment='center', horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def save_results(paths, results):
    result_folder = os.path.join(paths.tmp_root, 'results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        os.makedirs(os.path.join(result_folder, 'figs'))

    with open(os.path.join(result_folder, 'labels.p'), 'wb') as f:
        pickle.dump(results, f)


def load_results(paths):
    with open(os.path.join(paths.tmp_root, 'results', 'labels.p'), 'rb') as f:
        results = pickle.load(f)
    return results


def print_latex_table(data, row_labels, col_labels):
    data = data * 100
    row_labels = np.array(row_labels)
    row_labels = np.reshape(row_labels, [row_labels.shape[0], 1])
    data = np.hstack((row_labels, data))
    print
    print(tabulate.tabulate(data, tablefmt="latex", floatfmt=".1f", numalign="center", headers=col_labels))


def analyze_results(paths):
    def get_f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall)

    def format_table(predict_frame):
        data = np.empty((2, 8))
        data[0, 0:3] = 1.0/len(metadata.subactivities[:-1])
        data[0, 3] = get_f1_score(data[0, 0], data[0, 0])
        data[0, 4:7] = 1.0/len(metadata.affordances)
        data[0, 7] = get_f1_score(data[0, 4], data[0, 4])

        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_s[predict_frame], pred_s[predict_frame], labels=range(len(metadata.subactivities)-1), average='micro')
        data[1, 0] = precision
        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_s[predict_frame], pred_s[predict_frame], labels=range(len(metadata.subactivities)-1), average='macro')
        data[1, 1] = precision
        data[1, 2] = recall
        data[1, 3] = get_f1_score(precision, recall)

        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_u[predict_frame], pred_u[predict_frame], labels=range(len(metadata.affordances)), average='micro')
        data[1, 4] = precision
        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_u[predict_frame], pred_u[predict_frame], labels=range(len(metadata.affordances)), average='macro')
        data[1, 5] = precision
        data[1, 6] = recall
        data[1, 7] = get_f1_score(precision, recall)

        print_latex_table(data, methods, metrics)

    # ====================== Function starts here ======================
    # fig_folder = os.path.join(paths.tmp_root, 'results', 'figs')
    fig_folder = os.path.join(paths.project_root, 'fig', 'raw')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    seg_gt_s, seg_pred_s, seg_gt_u, seg_pred_u, gt_s, pred_s, gt_u, pred_u, gt_e, pred_e = load_results(paths)

    methods = ['chance', 'ours']
    metrics = ['P/R', 'Prec.', 'Recall', 'F1-score', 'P/R', 'Prec.', 'Recall', 'F1-score']
    # Evaluation
    # TODO: see if need to exclude "null" class
    # Online detection
    predict_frame = 0
    format_table(predict_frame)

    # Future detection
    predict_frame = 40
    for i in range(predict_frame):
        gt_s[predict_frame].extend(gt_s[i])
        pred_s[predict_frame].extend(pred_s[i])
        gt_u[predict_frame].extend(gt_u[i])
        pred_u[predict_frame].extend(pred_u[i])
    format_table(predict_frame)

    # Plot confusion matrices
    predict_frame = 0
    confusion_matrix = sklearn.metrics.confusion_matrix(gt_u[predict_frame], pred_u[predict_frame], labels=range(len(metadata.affordances)))
    plot_confusion_matrix(confusion_matrix, metadata.affordances, normalize=True, title='', filename=os.path.join(fig_folder, 'confusion_affordance.pdf'))

    confusion_matrix = sklearn.metrics.confusion_matrix(gt_s[predict_frame], pred_s[predict_frame], labels=range(len(metadata.subactivities) - 1))
    plot_confusion_matrix(confusion_matrix, metadata.subactivities[:-1], normalize=True, title='', filename=os.path.join(fig_folder, 'confusion_subactivity.pdf'))

    confusion_matrix = sklearn.metrics.confusion_matrix(gt_e, pred_e, labels=range(len(metadata.activities)))
    plot_confusion_matrix(confusion_matrix, metadata.activities, normalize=True, title='', filename=os.path.join(fig_folder, 'confusion_event.pdf'))


def main():
    paths = config.Paths()
    analyze_results(paths)
    pass


if __name__ == '__main__':
    main()
