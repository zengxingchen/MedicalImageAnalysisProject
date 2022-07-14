import random
import numpy as np
import torch


def Metric(labels, preds, soft=True, datatype='numpy', dim=2):
    import sklearn
    from scipy.special import softmax
    # labels = labels[:,1].reshape(-1,1)
    # preds = preds[:,1].reshape(-1,1)
    if soft == False:
        preds = softmax(preds, axis=1)
    # preds = output.max(1)[1]
    if datatype == 'numpy' and dim == 2:
        preds = (preds == preds.max(axis=1, keepdims=1)).astype(int)[:, 1]
    elif datatype == 'tensor' and dim == 2:
        preds = preds.max(1)[1].type_as(labels)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    # preds = preds[:, 1]
    # preds = torch.tensor(preds)
    # labels = torch.tensor(labels)
    result = []
    acc = sklearn.metrics.accuracy_score(labels, preds)
    auc = sklearn.metrics.roc_auc_score(labels, preds)
    f1 = sklearn.metrics.f1_score(labels, preds)
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    # recall = sklearn.metrics.recall_score(labels, preds)
    sen = cm[1][1] / (cm[1][1] + cm[1][0])
    spe = cm[0][0] / (cm[0][0] + cm[0][1])

    result.append(acc)
    result.append(auc)
    result.append(sen)
    result.append(spe)
    result.append(f1)

    return result


def Calc_All_Metric(result_list):
    ALL_ACC_list = []
    ALL_AUC_list = []
    ALL_SEN_list = []
    ALL_SPE_list = []
    ALL_F1_list = []
    times = 0
    for Best_result_list in result_list:
        ACC_list, AUC_list, SEN_list, SPE_list, F1_list = [], [], [], [], []
        for item in Best_result_list:
            ACC_list.append(item[0])
            AUC_list.append(item[1])
            SEN_list.append(item[2])
            SPE_list.append(item[3])
            F1_list.append(item[4])
        times += 1
        # print("=" * 30 + " {}-th result".format(times) + "=" * 30)
        # print("ACC: {:.4}% ± {:.4}%".format(np.mean(ACC_list)*100, np.std(ACC_list)*100))
        # print("AUC: {:.4}% ± {:.4}%".format(np.mean(AUC_list)*100, np.std(AUC_list)*100))
        # print("SEN: {:.4}% ± {:.4}%".format(np.mean(SEN_list)*100, np.std(SEN_list)*100))
        # print("SPE: {:.4}% ± {:.4}%".format(np.mean(SPE_list)*100, np.std(SPE_list)*100))
        # print("F1: {:.4}% ± {:.4}%".format(np.mean(F1_list)*100, np.std(F1_list)*100))
        ALL_ACC_list.append(np.mean(ACC_list))
        ALL_AUC_list.append(np.mean(AUC_list))
        ALL_SEN_list.append(np.mean(SEN_list))
        ALL_SPE_list.append(np.mean(SPE_list))
        ALL_F1_list.append(np.mean(F1_list))

    # 5此10折结果
    print("=" * 40 + " Final result " + "=" * 40)
    print("ACC: {:.4}%±{:.3}%".format(
        np.mean(ALL_ACC_list)*100, np.std(ALL_ACC_list)*100))
    print("AUC: {:.4}%±{:.3}%".format(
        np.mean(ALL_AUC_list)*100, np.std(ALL_AUC_list)*100))
    print("SEN: {:.4}%±{:.3}%".format(
        np.mean(ALL_SEN_list)*100, np.std(ALL_SEN_list)*100))
    print("SPE: {:.4}%±{:.3}%".format(
        np.mean(ALL_SPE_list)*100, np.std(ALL_SPE_list)*100))
    print("F1: {:.4}%±{:.3}%".format(
        np.mean(ALL_F1_list)*100, np.std(ALL_F1_list)*100))


def set_seed(seed=0):
    """_summary_
    在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，当得到比较
    好的结果时我们通常希望这个结果是可以复现的，在pytorch中，通过设置随机数种子也可以达到这么目的。
    Args:
        seed (int): Defaults to 0.
    """
    # For custom operators, you might need to set python seed as well:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # You can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
