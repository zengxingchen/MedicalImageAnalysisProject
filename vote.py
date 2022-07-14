import numpy as np
import argparse
from dataloader import load_data
from utils import Metric

# ============== 解析命令行参数 =============== #
parser = argparse.ArgumentParser()
parser.add_argument('setting', type=str, help='the setting of network')
args = parser.parse_args()

# ============== 硬投票 =============== #
#TODO 感觉硬投票有点问题 https://www.codenong.com/cs106310449/
labels = np.load('./result/true_labels.npy', allow_pickle=True)
pred_aal = np.load('./result/test_aal.npy', allow_pickle=True)
pred_cc200 = np.load('./result/test_cc200.npy', allow_pickle=True)
pred_dosenbach = np.load('./result/test_dosenbach160.npy', allow_pickle=True)


# 硬投票
all_result = pred_aal + pred_cc200 + pred_dosenbach
all_result = np.where(all_result > 1, 1, 0)


result = Metric(labels, all_result, soft=True, dim=1, datatype="numpy")

print("ACC: {:.2f}%".format(result[0] * 100))
print("AUC: {:.2f}%".format(result[1] * 100))
print("SEN: {:.2f}%".format(result[2] * 100))
print("SPE: {:.2f}%".format(result[3] * 100))
print("F1: {:.2f}%".format(result[4] * 100))

with open('lab_records.txt', 'a') as f:
  f.write("==================================\n")
  if args.setting:
    f.write("{}\n".format(args.setting))
  f.write("ACC: {:.2f}%\n".format(result[0] * 100))
  f.write("AUC: {:.2f}%\n".format(result[1] * 100))
  f.write("SEN: {:.2f}%\n".format(result[2] * 100))
  f.write("SPE: {:.2f}%\n".format(result[3] * 100))
  f.write("F1: {:.2f}%\n".format(result[4] * 100))
  f.write("==================================\n")