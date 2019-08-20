#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

# Evaluate
def accuracy(predictions, Y):
    """ calculate accuracy """
    error = predictions - Y
    #print(error)
    acc = sum(error == 0) / len(error)
    return acc

def precision_recall_f1score(predictions, Y_true):
    """ calculate precision_recall_f1score
    二分类，建议使用F1分数作为最终评估标准
    """
    precision, recall, f1score, _ = precision_recall_fscore_support(Y_true, 
                                        predictions, pos_label=1, average='binary')
    return precision, recall, f1score

def precision_recall_f1score_final(predictions, Y_true):
    """ calculate precision_recall_f1score
    三分类，建议使用UAR作为最终评估标准
    """
    UAPrecision, UARecall, UAF1score, _ = precision_recall_fscore_support(Y_true, predictions, average='macro')
    return UAPrecision, UARecall, UAF1score


def get_XY(merged_features_fp, label_fp):
    """ 读取数据和标签并将其数值化，返回数值化的Numpy矩阵。
        参数：
            merged_features_fp：汇总后的特征文件路径
            label_fp：标签文件路径
        如果给定的文件名是None则对应返回None。
    """
    if merged_features_fp:
        data_train = pd.read_csv(merged_features_fp, encoding='utf-8', index_col=0)
        X = data_train.values
    else:
        X = None

    if label_fp:
        label_train = pd.read_csv(label_fp, encoding='utf-8', index_col=0)
        dummies_label = pd.get_dummies(label_train['label'], prefix='label')
        Y = dummies_label.label_AD.values
    else:
        Y = None
    return X, Y


def save_predict_result(test_predict):
    """ 保存预测结果
    参数:
        test_predict : 预测结果，一个Numpy数组
    """
    label_fp = '../data/1_preliminary_list_test.csv'
    Y_test_df = pd.read_csv(label_fp, encoding='utf-8', index_col=0)
    Y_test_df['pred_value'] = test_pred
    Y_test_df.loc[(Y_test_df.pred_value == 1), 'label'] = 'AD'
    Y_test_df.loc[(Y_test_df.pred_value == 0), 'label'] = 'CTRL'
    Y_test_df.drop(columns=['pred_value'], inplace=True)
    Y_test_df.to_csv('./result.csv')


if __name__ == '__main__':
    # 载入汇总的特征和训练标签
    merged_features_fp = '../fusion/train/merged.csv'
    label_fp = '../data/1_preliminary_list_train.csv'
    X_train, Y_train = get_XY(merged_features_fp, label_fp)
    
    merged_features_fp = '../fusion/test/merged.csv'
    X_test, _ = get_XY(merged_features_fp, None)

    # 构建模型
    model = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('classification', LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10, 100], penalty='l2', solver='liblinear', cv=4))
    ])

    # 在训练集上训练并评估
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)

    print('Accuracy:', accuracy(Y_train, train_pred))
    print('Precision: %f, Recall:%f, F1-score:%f' % precision_recall_f1score(Y_train, train_pred))

    # 预测测试集标签，并保存预测结果
    test_pred = model.predict(X_test)
    save_predict_result(test_pred)

