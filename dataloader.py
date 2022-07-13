import numpy as np
import csv
import scipy.sparse as sp
phenotype = r"data/Phenotypic_V1_0b_preprocessed1.csv"


def load_data(atlas="aal"):
    subject_ids = np.loadtxt(r"data/subject_ids.txt", dtype=str)
    labels = get_subject_score(subject_ids, score='DX_GROUP')
    # adj = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_ids)
    num_subject = len(subject_ids)
    y = np.zeros([num_subject])
    for i in range(num_subject):
        y[i] = int(labels[subject_ids[i]])
    # raw data label is 1 or 2, we change it ro 0 or 1
    labels = y - 1
    raw_features = np.load(rf"data/{atlas}_871.npy")
    features = preprocess_features(raw_features)
    return features, labels


# get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict


def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
