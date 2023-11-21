import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics
import warnings
import torch
from train import Train
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


if __name__ == '__main__':

    auc, acc, pre, recall, f1, fprs, tprs = Train(directory='data',
                                                  epochs=args.epochs,#100
                                                  aggregator='GraphSAGE',  # 'GraphSAGE'
                                                  embedding_size=256,
                                                  layers=2,
                                                  dropout=args.dropout,
                                                  slope=0.2,  # LeakyReLU
                                                  lr=args.lr,
                                                  wd=args.weight_decay,
                                                  random_seed=args.seed
                                                   )

    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc), np.std(auc)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc), np.std(acc)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre), np.std(pre)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall), np.std(recall)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1), np.std(f1)))

    mean_fpr = np.linspace(0, 1, 10000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))

    std_tpr = np.std(tpr, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()