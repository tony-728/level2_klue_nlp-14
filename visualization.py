import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import pickle as pickle

from ast import literal_eval

def visualization_base(base_sheet, binary, pred, label, epoch_num, metrics, loss):
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label = pickle.load(f)
    
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
        
    print("Drawing Graph..")
    base_sheet = pd.read_csv(base_sheet)
    if binary:
        base_sheet["label"] = base_sheet["label"].apply(lambda x : "no_relation" if x == "no_relation" else "relation")
    
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
    if binary == False:
        for v in pred:
            num_label.append(dict_num_to_label[v])
    else:
        for v in pred:
            num_label.append("no_relation" if v == 0 else "relation")
            
    base_sheet["pred"] = num_label
    
    if binary == False:
        base_sheet["label"] = base_sheet["label"].apply(lambda x : dict_label[x])
        base_sheet["pred"] = base_sheet["pred"].apply(lambda x : dict_label[x])
    else:
        base_sheet["label"] = base_sheet["label"].apply(lambda x : 0 if x == "no_relation" else 1)
        base_sheet["pred"] = base_sheet["pred"].apply(lambda x : 0 if x == "no_relation" else 1)
    
    base_sheet = base_sheet[['label', 'pred', 'answer']]
    
    #base_sheet.to_csv(f"e{epoch_num}_F{round(metrics['micro f1 score'], 2)}_au{round(metrics['auprc'], 2)}_ac{round(metrics['accuracy'], 2)}_loss{round(loss, 2)}.csv", index=False)
    
    plt.figure(figsize=(20, 15))
    temp = []
        
    parallel_coordinates(base_sheet, 'answer', sort_labels=True, color=['#FE2E2E', '#2E2EFE'], alpha=0.1)
    
    plt.savefig(f"e{epoch_num}_F{round(metrics['micro f1 score'], 2)}_au{round(metrics['auprc'], 2)}_ac{round(metrics['accuracy'], 2)}_loss{round(loss, 2)}.png")
    plt.clf()