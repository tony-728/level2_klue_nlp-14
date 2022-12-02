import os

import torch
from torch.utils.data import Dataset

import pandas as pd

import pickle as pickle

# GPU가 여러개 일때는 병렬처리가 필요할 수 있지만
# 현재는 GPU가 하나이기 때문에 False로 변경해도 무방해보임
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Original_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, mode="train", data_mode="entity"):
        pd_dataset = pd.read_csv(data_path)
        raw_dataset = preprocessing_dataset(pd_dataset, data_mode=data_mode)
        raw_labels = raw_dataset["label"].values

        self.data = tokenize_dataset(raw_dataset, tokenizer)

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
        return item, labels

    def __len__(self):
        return len(self.labels)


class RE_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, mode="train", data_mode="entity"):
        pd_dataset = pd.read_csv(data_path)
        raw_dataset = preprocessing_dataset(pd_dataset, data_mode=data_mode)
        raw_labels = raw_dataset["label"].values

        self.data = tokenize_dataset(raw_dataset, tokenizer)

        self.marker_idx = get_marker_idx(
            self.data["input_ids"].tolist(), tokenizer, "@", "^"
        )  # subject marker : @ obj marker: ^

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
        markers = {key: val[idx] for key, val in self.marker_idx.items()}
        return item, labels, markers

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset, data_mode="entity"):
    subject_entity = []
    object_entity = []
    preprocessed_sentences = []
    task_sentence = []  # T5에게 줄 tast 문장

    for i, j, k in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        """
        if want to use entity's type
        subj_type = i['type']
        obj_type = j['type']
        """

        subj_dict = eval(i)
        obj_dict = eval(j)

        subject_entity.append(subj_dict["word"])
        object_entity.append(obj_dict["word"])

        ss = subj_dict["start_idx"]
        se = subj_dict["end_idx"]
        os = obj_dict["start_idx"]
        oe = obj_dict["end_idx"]

        if data_mode == "entity":
            if os < ss:
                preprocessed_sentences.append(
                    (
                        k[:os]
                        + " ^ "
                        + k[os : oe + 1]
                        + " ^ "
                        + k[oe + 1 : ss]
                        + " @ "
                        + k[ss : se + 1]
                        + " @ "
                        + k[se + 1 :]
                    )
                )
            else:
                preprocessed_sentences.append(
                    (
                        k[:ss]
                        + " @ "
                        + k[ss : se + 1]
                        + " @ "
                        + k[se + 1 : os]
                        + " ^ "
                        + k[os : oe + 1]
                        + " ^ "
                        + k[oe + 1 :]
                    )
                )

        elif data_mode == "type_entity":
            type_en_ko = {
                "PER": "사람",
                "ORG": "단체",
                "POH": "기타",
                "DAT": "날짜",
                "LOC": "장소",
                "NOH": "수량",
            }
            st = type_en_ko[subj_dict["type"]]
            ot = type_en_ko[obj_dict["type"]]

            # subject와 object의 관계
            # T5에서 사용할 문장
            # t_sentence = f"{subj_dict['word']}와 {obj_dict['word']}의 관계는 무엇인가?: "
            # task_sentence.append(t_sentence)

            if os < ss:
                preprocessed_sentences.append(
                    (
                        k[:os]
                        + " ^ ◇ "
                        + ot
                        + " ◇ "
                        + k[os : oe + 1]
                        + " ^ "
                        + k[oe + 1 : ss]
                        + " @ □ "
                        + st
                        + " □ "
                        + k[ss : se + 1]
                        + " @ "
                        + k[se + 1 :]
                    )
                )
            else:
                preprocessed_sentences.append(
                    (
                        k[:ss]
                        + " @ □ "
                        + st
                        + " □ "
                        + k[ss : se + 1]
                        + " @ "
                        + k[se + 1 : os]
                        + " ^ ◇ "
                        + ot
                        + " ◇ "
                        + k[os : oe + 1]
                        + " ^ "
                        + k[se + 1 :]
                    )
                )

    output_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            # "t_sentence": t_sentence,
            "sentence": preprocessed_sentences,
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )

    return output_dataset


def tokenize_dataset(dataset, tokenizer):

    tokenized_sentences = tokenizer(
        dataset["sentence"].tolist(),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )
    return tokenized_sentences


def get_marker_idx(input_ids_list, tokenizer, subj_marker, obj_marker):
    subj_marker_ids = int(
        tokenizer.encode(subj_marker, add_special_tokens=False, return_tensors="pt")
    )
    obj_marker_ids = int(
        tokenizer.encode(obj_marker, add_special_tokens=False, return_tensors="pt")[0][
            -1
        ]
    )
    marker_idx = {"ss": [], "se": [], "os": [], "oe": []}
    for input_ids in input_ids_list:
        tmp = input_ids
        subjs = [i for i, e in enumerate(tmp) if e == subj_marker_ids]
        if len(subjs) != 2:
            print(tokenizer.decode(torch.tensor(tmp)))
            # print(f"subj marker {subj_marker} is wrong value try another marker!")
            # exit()

        objs = [i for i, e in enumerate(tmp) if e == obj_marker_ids]
        if len(objs) != 2:
            print(tokenizer.decode(torch.tensor(tmp)))
            # print(f"obj marker {obj_marker} is wrong value try another marker!")
            # exit()

        # entity만 사용하도록 바꿈
        marker_idx["ss"].append(subjs[0] + 1)  # entity를 가리키게 됨
        marker_idx["se"].append(subjs[1])  # ^
        marker_idx["os"].append(objs[0] + 1)  # entity를 가리키게 됨
        marker_idx["oe"].append(objs[1])  # @

    return marker_idx


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
