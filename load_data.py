import os

import torch
from torch.utils.data import Dataset

import pandas as pd

import pickle as pickle


# GPU가 여러개 일때는 병렬처리가 필요할 수 있지만
# 현재는 GPU가 하나이기 때문에 False로 변경해도 무방해보임
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Dataset(Dataset):
    def __init__(self, data_path, tokenizer, mode="train"):
        pd_dataset = pd.read_csv(data_path)
        raw_dataset = preprocessing_dataset(pd_dataset)
        # raw_labels = raw_dataset["label"].values

        self.data = tokenize_dataset(raw_dataset, tokenizer, "item")
        self.labels =tokenize_dataset(raw_dataset, tokenizer, "label")

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        labels = {key: val[idx].clone().detach() for key, val in self.labels.items()}
        # labels = torch.tensor(self.labels[idx])
        return item, labels # same d

    def __len__(self):
        return len(self.labels)

def preprocessing_dataset(dataset):
    subject_entity = []
    object_entity = []
    preprocessed_sentences = []

    for i, j, k in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        subj_dict = eval(i)
        obj_dict = eval(j)

        subject_entity.append(subj_dict["word"])
        object_entity.append(obj_dict["word"])

        type_en_ko = {
            "PER": "사람",
            "ORG": "단체",
            "POH": "기타",
            "DAT": "날짜",
            "LOC": "장소",
            "NOH": "수량",
        }

        ss = subj_dict["start_idx"]
        se = subj_dict["end_idx"]
        os = obj_dict["start_idx"]
        oe = obj_dict["end_idx"]
        st = type_en_ko[subj_dict["type"]]
        ot = type_en_ko[obj_dict["type"]]
        
        task_frefix = '관계를 구분하세요 : '

        preprocessed_sentences.append(task_frefix + subj_dict['word'] + ' ' + obj_dict['word'] + ' ' + k)

        # if os < ss:
        #     preprocessed_sentences.append(
        #         (   task_frefix
        #             +
        #             k[:os]
        #             + " ^ ◇ "
        #             + ot
        #             + " ◇ "
        #             + k[os : oe + 1]
        #             + " ^ "
        #             + k[oe + 1 : ss]
        #             + " @ □ "
        #             + st
        #             + " □ "
        #             + k[ss : se + 1]
        #             + " @ "
        #             + k[se + 1 :]
        #         )
        #     )
        # else:
        #     preprocessed_sentences.append(
        #         (   task_frefix
        #             +
        #             k[:ss]
        #             + " @ □ "
        #             + st
        #             + " □ "
        #             + k[ss : se + 1]
        #             + " @ "
        #             + k[se + 1 : os]
        #             + " ^ ◇ "
        #             + ot
        #             + " ◇ "
        #             + k[os : oe + 1]
        #             + " ^ "
        #             + k[se + 1 :]
        #         )
        #     )
        
        """
        if want to use entity's type
        subj_type = i['type']
        obj_type = j['type']
        """

    output_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": preprocessed_sentences,
            # "subject_entity": subject_entity,
            # "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return output_dataset

def tokenize_dataset(dataset, tokenizer, type):

    if type == "item":
        tokenized_sentences = tokenizer(
            dataset['sentence'].tolist(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

    elif type == 'label':
        tokenized_sentences = tokenizer(
            dataset["label"].tolist(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

    return tokenized_sentences

def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset
