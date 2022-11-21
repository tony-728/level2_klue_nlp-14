import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np

from ast import literal_eval

def visualization_base(base_sheet, pred, label):
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
        
    print("그림 그리기 시작")
    base_sheet = pd.read_csv(base_sheet)
    
    pred = np.argmax(pred, axis=-1)
    
    #필요 없는 열 삭제
    base_sheet.drop('subject_entity', axis=1)
    base_sheet.drop('object_entity', axis=1)
    base_sheet.drop('id', axis=1)
    base_sheet.drop('sentence', axis=1)
    base_sheet.drop('source', axis=1)
    
    temp = []
    
    for i in range(len(pred)):
        if label[i] == pred[i]:
            temp.append("True")
        else:
            temp.append("False")
    
    base_sheet["answer"] = temp
    
    num_label = []
    for v in pred:
        num_label.append(dict_num_to_label[v])
    
    base_sheet["pred"] = num_label
    
    base_sheet = base_sheet[['label', 'pred', 'answer']]
    
    base_sheet.to_csv("test.csv", index=False)
    
    plt.figure(figsize=(10, 15))
    parallel_coordinates(base_sheet, 'answer', color=('r', 'b'), alpha=0.1)
    
    plt.savefig('test.png')
    plt.clf()