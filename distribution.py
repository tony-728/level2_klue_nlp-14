import pickle as pickle
import pandas as pd
import random


def get_sample_count(train_label_count):
    with open("data_distribution.pkl", "rb") as f:
        data_ratio = pickle.load(f)

    total_count = 32470
    train_ratio = dict()
    for label in train_label_count.keys():
        train_ratio[label] = train_label_count[label]/total_count * 100

    min_ratio = 100
    for k in data_ratio.keys():
        ratio = train_ratio[k] / data_ratio[k]
        if min_ratio > ratio:
            min_ratio = ratio

    count_list = dict()
    for k in data_ratio.keys():
        test_count = total_count * data_ratio[k] // 100
        count_list[k] = round(test_count * min_ratio)
    
    return count_list


def sample_data(source_path: str) -> pd.DataFrame:

    train_data = pd.read_csv(source_path)
    train_label_count=dict()
    for label in train_data['label']:
        train_label_count.setdefault(label, 0)
        train_label_count[label] += 1

    sampling_count = get_sample_count(train_label_count)

    label_data = dict()
    for key in sampling_count.keys():
        label_data.setdefault(key, [])

    for idx, row in train_data.iterrows():
        print(row)
        label_data[row['label']].append(row)

    sampled_data = pd.DataFrame([])
    
    for key in label_data.keys():
        df_ransample = pd.DataFrame(random.sample(label_data[key], sampling_count[key]))
        print(df_ransample)
        sampled_data = pd.concat([df_ransample, sampled_data])
    
    sampled_data.to_csv("renew_train.csv")
    return sampled_data
