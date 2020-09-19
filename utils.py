import numpy as np


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def calc_num_batches(total_num, batch_size):
    """
    Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.
    """
    return total_num // batch_size + int(total_num % batch_size != 0)


def cosine(s, t):
    return np.dot(s, t) / (np.linalg.norm(s) * np.linalg.norm(t))
