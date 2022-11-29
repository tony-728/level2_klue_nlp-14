import os

import torch
from torch.utils.data import Dataset

import pandas as pd
import random
import pickle as pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SP_Train_Dataset(Dataset):
    def __init__(self, data_path, tokenizer):
        pd_dataset = pd.read_csv(data_path)
        preprocessed_dataset = preprocessing_dataset(pd_dataset)
        self.question1, self.question2 = tokenize_dataset(preprocessed_dataset, tokenizer)
        self.answer1, self.answer2 = preprocessed_dataset['answer1'], preprocessed_dataset['answer2']
        self.labels = preprocessed_dataset['label']

    def __getitem__(self, idx):
        question1_pair = {key: value[idx].clone() for key, value in self.question1.items()}
        question1_pair['answer'] = self.answer1[idx]
        question2_pair = {key: value[idx].clone() for key, value in self.question2.items()}
        question2_pair['answer'] = self.answer2[idx]
        labels = self.labels[idx]

        return question1_pair, question2_pair, labels
        #return self.question1[idx], self.answer1, self.question2[idx], self.answer2, self.labels[idx]

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    answer1_list = []
    answer2_list = []
    question1_list = []
    question2_list = []
    for sentence, entity1_dict, entity2_dict, label in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity'], dataset['label']):
        entity1 = eval(entity1_dict)['word']
        entity1_s = eval(entity1_dict)['start_idx']
        entity1_e = eval(entity1_dict)['end_idx']
        entity2 = eval(entity2_dict)['word']
        entity2_s = eval(entity2_dict)['start_idx']
        entity2_e = eval(entity2_dict)['end_idx']

        if label == 'no_relation': 
            sentence1 = sentence
            sentence2 = sentence
        else:
            sentence1 = sentence[:entity2_s] + "@" + sentence[entity2_s:entity2_e+1] + "@" +sentence[entity2_e+1:] #entity2가 answer인 경우
            sentence2 = sentence[:entity1_s] + "@" + sentence[entity1_s:entity2_e+1] + "@" +sentence[entity1_e+1:]
        
        question1, question2 = generate_question(entity1, entity2, label)
        question1 = "[CLS]" + question1 + " [SEP] " + sentence1 + " [SEP]"
        question2 = "[CLS]" + question2 + " [SEP] " + sentence2 + " [SEP]"
        question1_list.append(question1)
        question2_list.append(question2)
        if label != 'no_relation':
            answer1_list.append(entity1)
            answer2_list.append(entity2)
        else:
            answer1_list.append('no_relation')
            answer2_list.append('no_relation')

    output_dataset = pd.DataFrame(
        {
        "id": dataset["id"],
        "question1": question1_list,
        "question2": question2_list,
        "answer1": answer1_list,
        "answer2": answer2_list,
        "label": dataset["label"]
        }
    )
    return output_dataset


def tokenize_dataset(dataset, tokenizer):
    tokenized_question1 = tokenizer(
        dataset["question1"].tolist(),
        return_tensors = "pt",
        padding = "max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )
    tokenized_question2 = tokenizer(
        dataset["question2"].tolist(),
        return_tensors = "pt",
        padding = "max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )
    #dataset['question1'] = tokenized_question1
    #dataset['question2'] = tokenized_question2
    return tokenized_question1, tokenized_question2



def generate_question(entity1, entity2, label):
    label2question = {
        "per:other_family": ["의 가족은 누구입니까?","의 가족은 누구입니까?"],
        "per:spouse": ["의 배우자는 누구입니까?", "의 배우자는 누구입니까?"],
        "per:schools_attended": ["가 다닌 학교는 무엇입니까?", "에 다닌 사람은 누구입니까?"],
        "per:date_of_death": ["가 죽은 날짜는 언제입니까?", "에 죽은 사람은 누구입니까?"],
        "per:date_of_birth": ["가 태어난 날짜는 언제입니까?", "에 태어난 사람은 누구입니까?"],
        "per:title": ["의 역할은 무엇입니까?", " 역할인 사람은 누구입니까?"],
        "per:parents": ["의 부모는 누구입니까?", "의 자식은 누구입니까?"],
        "per:children": ["의 자식은 누구입니까?", "의 부모는 누구입니까?"],
        "per:origin": ["의 출신은 무엇입니까?", "출신은 누구입니까?"],
        "per:employee_of": ["가 일하는 장소는 어디입니까?", "에서 일하는 사람은 누구입니까?"],
        "per:siblings": ["의 형제자매는 누구입니까?","의 형제자매는 누구입니까?"],
        "per:alternate_names": ["의 다른 이름은 무엇입니까?", "의 다른 이름은 무엇입니까?"],
        "per:religion": ["의 종교는 무엇입니까?", "를 따르는 사람은 누구입니까?"],
        "per:place_of_residence":["가 거주하는 장소는 어디입니까?","에 거주하는 사람은 누구입니까?"],
        "per:place_of_death": ["가 죽은 장소는 어디입니까?", "에서 죽은 사람은 누구입니까?"],
        "per:place_of_birth": ["가 태어난 장소는 어디입니까?", "에서 태어난 사람은 누구입니까?"],
        "per:product": ["의 상품은 무엇입니까?", "의 제작자는 누구입니까?"],
        "per:colleagues": ["의 동료는 누구입니까?", "의 동료는 누구입니까?"],
        "org:members": ["의 하위 기관은 무엇입니까?", "의 상위 기관은 무엇입니까?"],
        "org:member_of": ["가 소속된 것은 무엇입니까?", "에 소속된 것은 무엇입니까?"],
        "org:number_of_employees/members": ["의 구성원은 몇 개입니까?", "를 개수로 가지는 것은 무엇입니까?"],
        "org:political/religious_affiliation": ["의 정치적,종교적 요소는 무엇입니까?", "가 정치적,종교적 요소로 작용하는 것은 무엇입니까?"],
        "org:founded": ["이 설립된 것은 언제입니까?", "에 설립된 것은 무엇입니까?"],
        "org:founded_by": ["을 설립한 것은 누구입니까?", "가 설립한 것은 무엇입니까?"],
        "org:dissolved": ["이 해체된 것은 언제입니까?", "에 해체된 것은 무엇입니까?"],
        "org:top_members/employees": ["의 대표자는 누구입니까?", "가 대표자인 것은 무엇입니까?"],
        "org:alternate_names": ["의 다른 이름은 무엇입니까?", "의 다른 이름은 무엇입니까?"],
        "org:place_of_headquarters": ["의 본부는 어디입니까?", "의 산하는 무엇입니까?"],
        "org:product": ["의 제품은 무엇입니까?", "를 생산하는 건 무엇입니까?"]
    }

    if label != "no_relation":
        question1 = entity2 + label2question[label][0]
        question2 = entity1 + label2question[label][1]
    else:
        question1 = entity2 + label2question[random.choice(list(label2question.keys()))][0]
        question2 = entity1 + label2question[random.choice(list(label2question.keys()))][1]
    
    return question1, question2
        
