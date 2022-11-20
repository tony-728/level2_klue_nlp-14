import pickle as pickle
import torch
from torch.utils.data import Dataset
import pandas as pd


class RE_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, mode="train"):
        pd_dataset = pd.read_csv(data_path)
        raw_dataset = preprocessing_dataset(pd_dataset)
        raw_labels = raw_dataset["label"].values

        self.data = tokenize_dataset(raw_dataset, tokenizer)
        
        self.marker = get_marker_idx(self.data['input_ids'].tolist(), tokenizer, '@', '^') #subject marker : @ obj marker: ^
        if mode == "train":
            self.labels = label_to_num(raw_labels)
        elif mode == "prediction":
            self.labels = list(map(int, raw_labels))
        else:
            print("check your mode")
            exit()

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        labels = torch.tensor(self.labels[idx])
        marker_idx = {key: val[idx].clone().detach() for key, val in self.marker.items()}
        return item, labels, marker_idx

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]
        subject_entity.append(i)
        object_entity.append(j)
    output_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return output_dataset


def tokenize_dataset(dataset, tokenizer):
    new_sentence = []
    for sent, subj, obj in zip(dataset["sentence"], dataset["subject_entity"], dataset["object_entity"]):
        tmp = ""
        tmp = sent.replace(subj[2:-1], ' @ ' + subj[2:-1] + ' @ ')
        tmp = tmp.replace(obj[2:-1], ' ^ ' + obj[2:-1] + ' ^ ')
        new_sentence.append(tmp)
    
    tokenized_sentences = tokenizer(
        new_sentence,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 256,
        add_special_tokens = True
    )
    return  tokenized_sentences

def get_marker_idx(input_ids_list, tokenizer, subj_marker, obj_marker):
    subj_marker_ids = int(tokenizer.encode(subj_marker, add_special_tokens = False, return_tensors = "pt"))
    obj_marker_ids = int(tokenizer.encode(obj_marker, add_special_tokens = False, return_tensors = "pt"))
    marker_idx = {
        'ss' : [],
        'se' : [],
        'os' : [],
        'oe': []}
    for input_ids in input_ids_list:
        tmp = input_ids
        subjs = [i for i,e in enumerate(tmp) if e == subj_marker_ids]
        if len(subjs) != 2:
            print(tokenizer.decode(torch.tensor(tmp)))
            #print(f"subj marker {subj_marker} is wrong value try another marker!")
            #exit()
        
        objs = [i for i,e in enumerate(tmp) if e == obj_marker_ids]
        if len(objs) != 2:
            print(tokenizer.decode(torch.tensor(tmp)))
            #print(f"obj marker {obj_marker} is wrong value try another marker!")
            #exit()
        marker_idx['ss'].append(subjs[0])
        marker_idx['se'].append(subjs[1])
        marker_idx['os'].append(objs[0])
        marker_idx['oe'].append(objs[1])

    return marker_idx

"""
def tokenize_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences
"""

def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label
