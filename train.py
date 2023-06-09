import numpy as np
import pandas as pd
import math
import argparse
# import tqdm
# import gpytorch
# from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp.autocast_mode as autocast
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from seq_load_one_hot_NCP_EIIP import *
# from resnetwithCBAM import *
from model_one_hot_NCP_EIIP import *

from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is {}'.format(device))

def save_checkpoint(state, is_best, model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'train_result' + '/' + 'checkpoint.pth.tar')
        # torch.save(state, '/home/li/public/lxj/one_hot/mr/51/checkpoint_mr.pth.tar')  #马

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn, 'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i]) + '\t' + str(y_pred[i]) + '\n')


def calculate_metric(gt, pred):
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)
    # print('Sensitivity:', TP / float(TP + FN))
    # print('Specificity:', TN / float(TN + FP))


if __name__ == '__main__':

    torch.manual_seed(1000)

    parser = argparse.ArgumentParser()

    # main option
    parser.add_argument("-pos_fa", "--positive_fasta", action="store", dest='pos_fa', required=True,
                        help="positive fasta file")
    parser.add_argument("-neg_fa", "--negative_fasta", action="store", dest='neg_fa', required=True,
                        help="negative fasta file")

    parser.add_argument("-outdir", "--out_dir", action="store", dest='out_dir', required=True,
                        help="output directory")

    # rnn option
    parser.add_argument("-rnntype", "--rnn_type", action="store", dest='rnn_type', default='LSTM', type=str,
                        help="[capital] LSTM(default), GRU")
    parser.add_argument("-hidnum", "--hidden_num", action="store", dest='hidden_num', default=128, type=int,
                        help="rnn size")
    parser.add_argument("-rnndrop", "--rnn_drop", action="store", dest='rnn_drop', default=0.5, type=float,
                        help="rnn size")

    # fc option
    parser.add_argument("-fcdrop", "--fc_drop", action="store", dest='fc_drop', default=0.5, type=float,
                        help="Optional: 0.5(default), 0~0.5(recommend)")

    # optimization option
    parser.add_argument("-optim", "--optimization", action="store", dest='optim', default='Adam', type=str,
                        help="Optional: Adam(default), ")
    parser.add_argument("-epochs", "--max_epochs", action="store", dest='max_epochs', default=15, type=int,
                        help="max epochs")
    parser.add_argument("-lr", "--learning_rate", action="store", dest='learning_rate', default=0.0001, type=float,
                        help="Adam: 0.0001(default), 0.0001~0.01(recommend)")
    # parser.add_argument("-lrstep", "--lr_decay_step", action="store", dest='lr_decay_step', default=10, type=int,
    #                     help="learning rate decay step")
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=8, type=int,
                        help="batch size")

    args = parser.parse_args()

    model_path = '.'

    wordvec_len = 8
    HIDDEN_NUM = args.hidden_num

    LAYER_NUM = 3

    RNN_DROPOUT = args.rnn_drop
    FC_DROPOUT = args.fc_drop
    CELL = args.rnn_type
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size

    tprs = []
    ROC_aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    PR_aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    # pos_train_fa = 'chhit_80%_pos.fasta'
    # neg_train_fa = 'neg_samecdhitpos.fasta'
    # seqfeatures_pos_train = 'pos_CKSNAP_DAC_kmer_ANF.tsv'
    # seqfeatures_neg_train = 'neg_CKSNAP_DAC_kmer_ANF.tsv'

    # pos_train_fa = 'cdhit80%_pos.fasta'
    # neg_train_fa = 'cut51_neg_88746.fasta'

    pos_train_fa = args.pos_fa
    neg_train_fa = args.neg_fa
    # pos_train_fa = 'pos_ceshi111.fasta'
    # neg_train_fa = 'neg_ceshi111.fasta'

    # X_train, y_train, X_test, y_test = load_train_val_bicoding(pos_train_fa, neg_train_fa, seqfeatures_pos_train, seqfeatures_neg_train, BATCH_SIZE*2)
    # X_train, y_train, X_test, y_test = load_train_val_bicoding(pos_train_fa, neg_train_fa)
    X, y = load_train_val_bicoding(pos_train_fa, neg_train_fa)
    folds = StratifiedKFold(n_splits=5).split(X, y)
    for trained, valided in folds:
        X_train, y_train = X[trained], y[trained]
        X_test, y_test = X[valided], y[valided]
        X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test)
        X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)

        # model = BiLSTM_Attention(wordvec_len, HIDDEN_NUM , LAYER_NUM, DROPOUT)
        # model = CNN_RNN(HIDDEN_NUM, LAYER_NUM, DROPOUT, CELL)
        model = CNN51_RNN(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, RNN_DROPOUT, CELL)
        # model = ronghe(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, CELL)
        # model = nn.DataParallel(model)
        # model = resnet18CBAM()
        model = model.to(device)
        loss = torch.nn.CrossEntropyLoss(reduction='sum')
        # loss = loss.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if args.optim == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # scaler = torch.cuda.amp.GradScaler()

        best_acc = 0
        best_train_accuracy_score = 0
        patience = 0


        def train(model, loss, optimizer, x, y):

            model.train()

            # Reset gradient
            optimizer.zero_grad()

            # Forward
            # with torch.cuda.amp.autocast():
            fx = model(x)
            loss = loss.forward(fx, y)

            pred_prob = F.log_softmax(fx, dim=1)

            # Backward
            loss.backward()
            # scaler.scale(loss).backward()

            # grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # for p in model.parameters():
            #     p.data.add_(-LEARNING_RATE, p.grad.data)

            # Update parameters
            optimizer.step()
            # scaler.step(optimizer)

            # scaler.update()

            return loss.cpu().item(), pred_prob, list(np.array(y.cpu())), list(fx.data.cpu().detach().numpy().argmax(axis=1))  # cost,pred_probability and true y value


        def predict(model, x):
            model.eval()  # evaluation mode do not use drop out

            with torch.no_grad():
                fx = model(x)
            return fx


        EPOCH = args.max_epochs
        n_classes = 2
        n_examples = len(X_train)

        for i in range(EPOCH):
            start_time = time.time()

            cost = 0.
            y_pred_prob_train = []
            y_batch_train = []
            y_batch_pred_train = []

            num_batches = n_examples // BATCH_SIZE
            for k in range(num_batches):
                start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
                # X_train[start:end] , y_train[start:end] = X_train[start:end].to(device), y_train[start:end].to(device)
                output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end],
                                                                         y_train[start:end])
                cost += output_train
                # print(y_pred_prob.shape)

                prob_data = y_pred_prob.cpu().detach().numpy()
                # print(prob_data.shape)
                # if args.if_bce == 'Y':
                #     for m in range(len(prob_data)):
                #         y_pred_prob_train.append(prob_data[m][0])

                # else:
                for m in range(len(prob_data)):
                    # print(np.exp(prob_data)[m])
                    y_pred_prob_train.append(np.exp(prob_data)[m][1])

                # print(y_pred_prob_train)
                y_batch_train += y_batch
                y_batch_pred_train += y_pred_train

            scheduler.step()

            # rest samples
            start, end = num_batches * BATCH_SIZE, n_examples
            output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end],
                                                                     y_train[start:end])
            cost += output_train

            prob_data = y_pred_prob.cpu().detach().numpy()
            # print(prob_data)
            # if args.if_bce == 'Y':
            #     for m in range(len(prob_data)):
            #         y_pred_prob_train.append(prob_data[m][0])

            for m in range(len(prob_data)):
                y_pred_prob_train.append(np.exp(prob_data)[m][1])
            # print(y_pred_prob_train)

            y_batch_train += y_batch
            y_batch_pred_train += y_pred_train
            # print(type(y_batch_train[0]), type(y_pred_prob_train[0]), type(y_batch_pred_train[0]))
            # print(y_batch_train[0:10], y_pred_prob_train[0:10], y_batch_pred_train[0:10])

            # train AUC
            fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)


            train_accuracy_score = sklearn.metrics.accuracy_score(y_batch_train, y_batch_pred_train)
            train_recall_score = sklearn.metrics.recall_score(y_batch_train, y_batch_pred_train)
            train_precision_score = sklearn.metrics.precision_score(y_batch_train, y_batch_pred_train)
            train_f1_score = sklearn.metrics.f1_score(y_batch_train, y_batch_pred_train)
            train_mcc = sklearn.metrics.matthews_corrcoef(y_batch_train, y_batch_pred_train)

            # predict test
            fx_test = predict(model, X_test)
            y_pred_prob_test = []

            # if args.if_bce == 'Y':
            #     y_pred_test = []
            #     prob_data = F.sigmoid(output_test).data.numpy()
            #     for m in range(len(prob_data)):
            #         y_pred_prob_test.append(prob_data[m][0])
            #         if prob_data[m][0] >= 0.5:
            #             y_pred_test.append(1)
            #         else:
            #             y_pred_test.append(0)
            # else:
            y_pred_test = fx_test.cpu().data.numpy().argmax(axis=1)
            prob_data = F.log_softmax(fx_test, dim=1).data.cpu().numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(np.exp(prob_data)[m][1])

            #
            # print(type(y_batch_train), type(y_batch_pred_train), type(y_test), type(y_pred_prob_test))
            # print(y_test, len(y_test))
            # print(list(y_test))
            # print('+++++++++++++++')
            # test AUROC
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)


            test_specificity = calculate_metric(y_test, y_pred_test)
            test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
            test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
            test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
            test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
            test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)

            end_time = time.time()
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)

            print("Epoch %d, cost = %f, AUROC_train = %0.4f, acc = %.2f%%, AUROC_test = %0.4f ,train_accuracy = %0.4f, train_recall(sn) = %0.4f, train_precision = %0.4f, train_f1_score = %0.4f, train_mcc = %0.4f, test_accuracy = %0.4f, test_recall(sn) = %0.4f ,test_sp = %0.4f, test_precision = %0.4f, test_f1_score = %0.4f, test_mcc = %0.4f"
                % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), 100. * np.mean(y_pred_test == y_test),
                   auc(fpr_test, tpr_test), train_accuracy_score, train_recall_score, train_precision_score, train_f1_score,
                   train_mcc, test_accuracy_score, test_recall_score, test_specificity, test_precision_score, test_f1_score, test_mcc))

            # print("Epoch %d, cost = %f, AUROC_train = %0.3f, acc = %.2f%%, AUROC_test = %0.3f ,train_accuracy_score = %0.3f, train_recall_score = %0.3f, train_precision_score = %0.3f, train_f1_score = %0.3f, train_mcc = %0.3f"
            #     % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), 100. * np.mean(y_pred_test == y_test),
            #        auc(fpr_test, tpr_test), train_accuracy_score, train_recall_score, train_precision_score, train_f1_score,
            #        train_mcc))
            print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            cur_acc = 100. * np.mean(y_pred_test == y_test)
            # cur_train_accuracy_score = train_accuracy_score
            # is_best = bool(cur_train_accuracy_score > best_train_accuracy_score)
            is_best = bool(cur_acc > best_acc)
            # best_train_accuracy_score = max(cur_train_accuracy_score, best_train_accuracy_score)
            best_acc = max(cur_acc, best_acc)
            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best, model_path)

            # patience
            if not is_best:
                patience += 1
                if patience >= 5:
                    break

            else:
                patience = 0

            if is_best:
                # ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                #                     '/home/li/public/lxj/one_hot/mr/51/predout_train.tsv') #马
                ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                                    '/home/li/public/lxj/Dcirc_complement/layer_num/train_result/predout_train.tsv')

                # ytest_ypred_to_file(y_test, y_pred_prob_test,
                #                     '/home/li/public/lxj/one_hot/mr/51/predout_val.tsv') #马
                ytest_ypred_to_file(y_test, y_pred_prob_test,
                                    '/home/li/public/lxj/Dcirc_complement/layer_num/train_result/predout_val.tsv')

                # fpr_test_best, tpr_test_best, thresholds_test_best = roc_curve(y_test, y_pred_prob_test)
                # precision_test_best, recall_test_best, _ = precision_recall_curve(y_test, y_pred_prob_test)
                #

        fprArray.append(fpr_test)
        tprArray.append(tpr_test)
        thresholdsArray.append(thresholds_test)
        tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr_test, tpr_test)
        ROC_aucs.append(roc_auc)

        recall_array.append(recall_test)
        precision_array.append(precision_test)
        precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
        pr_auc = auc(recall_test ,precision_test)
        PR_aucs.append(pr_auc)


    colors = cycle(['#caffbf', '#ffc6ff' ,'#ffadad', '#ffd6a5', '#caffbf', '#9bf6ff', '#a0c4ff', '#bdb2ff'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(fprArray)), colors):
        plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color,
                 label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#c4c7ff',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ROC_mean_auc = auc(mean_fpr, mean_tpr)
    ROC_std_auc = np.std(ROC_aucs)
    # plt.plot(mean_fpr, mean_tpr, color='#ea7317',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (ROC_mean_auc, ROC_std_auc),
    #          lw=1.5, alpha=.9)
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('/home/li/public/lxj/Dcirc_complement/layer_num/train_result/roc_hela_test.png')
    # plt.show()
    plt.close(0)

    fig = plt.figure(1)
    for i, color in zip(range(len(recall_array)), colors):
        plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
                 label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    PR_mean_auc = auc(mean_recall, mean_precision)
    PR_std_auc = np.std(PR_aucs)

    # plt.plot(mean_recall, mean_precision, color='#ea7317',
    #          label=r'Mean PRC (AUPRC = %0.2f $\pm$ %0.2f)' % (PR_mean_auc, PR_std_auc),
    #          lw=1.5, alpha=.9)
    # std_precision = np.std(precisions, axis=0)
    # precision_upper = np.minimum(mean_precision + std_precision, 1)
    # precision_lower = np.maximum(mean_precision - std_precision, 0)
    # plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig('/home/li/public/lxj/Dcirc_complement/layer_num/train_result/pr_hela_test.png')
    # plt.show()
    plt.close(0)

    print('> best acc:', best_acc)


    def plot_prc_CV(data, label_column=0, score_column=1):
        precisions = []
        PR_aucs = []
        recall_array = []
        precision_array = []
        mean_recall = np.linspace(0, 1, 100)
        df = pd.read_csv(data, header=None, sep='\t')

        # for i in range(len(data)):
        precision, recall, _ = precision_recall_curve(np.array(df.iloc[:, label_column]),
                                                      np.array(df.iloc[:, score_column]))
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        pr_auc = auc(recall, precision)
        PR_aucs.append(pr_auc)

colors = cycle(['#5f0f40', '#9a031e' ,'#fb8b24', '#e36414', '#0f4c5c'])
fig, ax = plt.subplots(1, 1)

axins = ax.inset_axes((0.6, 0.4, 0.3, 0.3))


for i, color in zip(range(len(fprArray)), colors):
    ax.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color, label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))
    axins.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color, label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")



# 设置放大区间

# X轴的显示范围
xlim0 = 0.0
xlim1 = 0.05

# Y轴的显示范围

ylim0 = 0.91
ylim1 = 0.96

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)


tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

# 画两条线
xy = (xlim1,ylim1)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim0,ylim0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

plt.savefig(model_path + '/' + 'train_result' + '/' + 'big_roc.png', dpi=600)
plt.close(0)





fig, ax = plt.subplots(1, 1)
axins = ax.inset_axes((0.6, 0.3, 0.3, 0.3))

for i, color in zip(range(len(recall_array)), colors):
    ax.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
             label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))
    axins.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
             label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")

# X轴的显示范围
xlim0 = 0.95
xlim1 = 1.0

# Y轴的显示范围

ylim0 = 0.91
ylim1 = 0.96

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)


tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

# 画两条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

plt.savefig(model_path + '/' + 'train_result' + '/' + 'big_pr.png', dpi=600)
plt.close(0)
