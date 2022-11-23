import sklearn
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def compute_loss(pred, labels):
    loss = nn.CrossEntropyLoss()
    return loss(pred, labels)


def compute_metrics(pred, labels, so_combine):
    """validation을 위한 metrics function"""
    # labels = pred.label_ids

    # 원래 있던 형태
    # preds = pred.predictions.argmax(-1)
    # probs = pred.predictions

    preds = pred.argmax(-1)
    probs = pred

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels, so_combine)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def klue_re_micro_f1(preds, labels, so_combine = 0):
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )


def klue_re_auprc(probs, labels, so_combine):
    """KLUE-RE AUPRC (with no_relation)"""
    # probs = probs.detach().numpy()
    with open(f'/opt/ml/level2_klue_nlp-14/recent_pkl/{so_combine}_label2num.pkl', 'rb') as f:
        data = pickle.load(f)
    label_count = len(data)
    labels = np.eye(label_count)[labels]
    score = np.zeros((label_count,))
    for c in range(label_count):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0
