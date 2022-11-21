import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np

from ast import literal_eval

def str2dic(x):
    if type(x) == str:
        return literal_eval(x)

def visualization_base(base_sheet, pred, label):
    print("그림 그리기 시작")
    base_sheet = pd.read_csv(base_sheet)
    
    pred = np.argmax(pred, axis=-1)
    
    base_sheet["subject_entity"] = base_sheet["subject_entity"].apply(lambda x : str2dic(x))
    base_sheet["object_entity"] = base_sheet["object_entity"].apply(lambda x : str2dic(x))
    
    base_sheet["subject_type"] = base_sheet["subject_entity"].apply(lambda x : x["type"])
    base_sheet["object_type"] = base_sheet["object_entity"].apply(lambda x : x["type"])
    
    #필요 없는 열 삭제
    base_sheet.drop('subject_entity', axis=1)
    base_sheet.drop('object_entity', axis=1)
    base_sheet.drop('id', axis=1)
    base_sheet.drop('sentence', axis=1)
    base_sheet.drop('source', axis=1)
    
    base_sheet["pred"] = pred
    
    temp = []
    
    for i in range(len(base_sheet["pred"])):
        if label[i] == base_sheet["pred"][i]:
            temp.append("True")
        else:
            temp.append("False")
    
    base_sheet["answer"] = temp
    
    base_sheet.drop('pred', axis=1)
    
    base_sheet = base_sheet[['subject_type', 'object_type', 'label', 'answer']]
    
    base_sheet.to_csv("test.csv", index=False)
    
    plt.figure(figsize=(10, 15))
    parallel_coordinates(base_sheet, 'answer', color=('r', 'b'), alpha=0.1)
    
    plt.savefig('test.png')
    plt.clf()