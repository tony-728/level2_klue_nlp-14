import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import pickle as pickle

from ast import literal_eval

def visualization_base(base_sheet, pred, label, epoch_num, metrics, loss):
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
        
    print("Drawing Graph..")
    base_sheet = pd.read_csv(base_sheet)
    
    pred = np.argmax(pred, axis=-1)
    
    temp = []
    
    #0이 True, 1이 False. 색깔이 맞지 않아서 일부러 반대로 설정해주었음.
    for i in range(len(pred)):
        if label[i] == pred[i]:
            temp.append(0)
        else:
            temp.append(1)
    
    base_sheet["answer"] = temp
    
    num_label = []
    for v in pred:
        num_label.append(dict_num_to_label[v])
    
    base_sheet["pred"] = num_label
    
    base_sheet = base_sheet[['label', 'pred', 'answer']]
    
    #base_sheet.to_csv(f"e{epoch_num}_F{round(metrics['micro f1 score'], 2)}_au{round(metrics['auprc'], 2)}_ac{round(metrics['accuracy'], 2)}_loss{round(loss, 2)}.csv", index=False)
    
    plt.figure(figsize=(20, 15))
    parallel_coordinates(base_sheet, 'answer', sort_labels=True, color=['#FE2E2E', '#2E2EFE'], alpha=0.1)
    
    plt.savefig(f"e{epoch_num}_F{round(metrics['micro f1 score'], 2)}_au{round(metrics['auprc'], 2)}_ac{round(metrics['accuracy'], 2)}_loss{round(loss, 2)}.png")
    plt.clf()