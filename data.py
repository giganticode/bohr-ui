import re
import subprocess
from typing import List, Dict

import jsonlines as jsonlines
import streamlit as st

import pandas as pd
import numpy as np
import dvc.api
from dvc.exceptions import PathMissingError

from tqdm import tqdm

from config import dataset_id_to_mnemonic, dataset_mnemonic_to_id
from vcs import get_path_to_revision


bohr_bugginess_repo = 'https://github.com/giganticode/bohr-workdir-bugginess'
diff_classifier_repo = 'https://github.com/giganticode/diff-classifier'

CHUNK_SIZE = 10000
TRANSFORMER_REGEX = re.compile('fine_grained_changes_transformer_(\\d+)')
KEYWORD_REGEX = re.compile('bug.*_message_keyword_(.*)')


@st.cache(show_spinner=False)
def get_label_matrix_locally(tmp_path):
    with st.spinner(f'Loading label matrix from {tmp_path}'):
        return pd.read_pickle(tmp_path)


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_label_matrix(dataset_name: str) -> pd.DataFrame:
    path = f'runs/bugginess/dataset_debugging/{dataset_name}/heuristic_matrix.pkl'
    path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', False)
    with st.spinner(f'Loading label matrix for dataset `{dataset_name}`'):
        with st.spinner(f'Reading `{path}` from `{bohr_bugginess_repo}`'):
            subprocess.run(["dvc", "pull", path], cwd=path_to_revision)
    return get_label_matrix_locally(f'{path_to_revision}/{path}')


@st.cache(allow_output_mutation=True, show_spinner=False)
def read_labeled_dataset(model, selected_dataset):
    with st.spinner(f'Loading labels by model `{model}` for dataset `{selected_dataset}`'):
        try:
            df = read_labeled_dataset_from_bohr(model, selected_dataset)
        except (PathMissingError, FileNotFoundError):
            df = read_labeled_dataset_from_transformers(model, selected_dataset)
        return df


@st.cache(show_spinner=False)
def read_labeled_dataset_from_transformers(model, selected_dataset):
    label_to_int = {'NonBugFix': 0, 'BugFix': 1}
    path = f'models/{model}/assigned_labels_{selected_dataset}.csv'
    with st.spinner(f'Reading `{path}` from `{diff_classifier_repo}`'):
        with dvc.api.open(path, repo=diff_classifier_repo) as f:
            df = pd.read_csv(f).rename(columns={'true_label': 'label'}, inplace=False)
    if 'label' in df.columns:
        df.loc[:, 'label'] = df.apply(lambda row: label_to_int[row["label"]], axis=1)
    else:
        df.loc[:, 'label'] = 1
    df.loc[:,  'prob_CommitLabel.BugFix'] = df.apply(lambda row: (row['probability'] if row['prediction'] == 'BugFix' else 1 - row['probability']), axis=1)
    return df


@st.cache(show_spinner=False)
def read_labeled_dataset_from_bohr(model, selected_dataset):
    label_to_int = {'CommitLabel.NonBugFix': 0, 'CommitLabel.BugFix': 1}
    path = f'runs/bugginess/{model}/{selected_dataset}/labeled.csv'
    path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', False)

    subprocess.run(["dvc", "pull", path], cwd=path_to_revision)
    full_path = f'{path_to_revision}/{path}'
    with st.spinner(f'Reading `{full_path}` from `{bohr_bugginess_repo}`'):
        print(f'Reading labeled dataset from bohr repo at: {full_path}')
        with open(full_path) as f:
            df = pd.read_csv(f)
    if 'label' in df.columns:
        df.loc[:, 'label'] = df.apply(lambda row: label_to_int[row["label"]], axis=1)
    else:
        df.loc[:, 'label'] = 1.0
    return df


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_dataset_chunk(dataset, chunk: int) -> List[Dict]:
    path = f'cached-datasets/{dataset}.jsonl'
    with st.spinner(f"Loading dataset `{path}` from `{bohr_bugginess_repo}`"):
        path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', False)
        subprocess.run(["dvc", "pull", path], cwd=path_to_revision)
        return get_dataset_from_path(f'{path_to_revision}/{path}', chunk)


def get_dataset_from_path(tmp_path, chunk):
    dataset = []
    with jsonlines.open(tmp_path) as reader:
        try:
            reader_iter = iter(reader)
            print(f'Skipping {chunk * CHUNK_SIZE} data points')
            for i in range(chunk * CHUNK_SIZE):
                next(reader_iter)
            print("Loading dataset ...")
            for i in tqdm(range(CHUNK_SIZE)):
                commit = next(reader_iter)
                dataset.append(commit)
        except StopIteration:
            pass
    return dataset


def get_dataset(dataset: str, batch=0):
    return get_dataset_chunk(dataset, batch)


def compute_lf_coverages(d):
    lf_values = [0, 1]
    resres = []
    for lf_value in lf_values:
        res = []
        for dataset in d:
            df = get_label_matrix(dataset)
            ln = len(df)
            r = (df == lf_value).sum(axis=0) / float(ln)
            res.append(r)
        cc = pd.concat(res, axis=1)

        def decompose_heuristic_name(s):
            matcher = KEYWORD_REGEX.fullmatch(s)
            if matcher:
                return 'keyword', matcher.group(1)
            else:
                matcher = TRANSFORMER_REGEX.fullmatch(s)
                if matcher:
                    return 'transformer', f'{int(matcher.group(1))+10}>p>{matcher.group(1)}'
                else:
                    return 'file metric', s

        def process_index(s):
            return *decompose_heuristic_name(s), lf_value

        cc.index = pd.MultiIndex.from_tuples([process_index(k) for k,v in cc.iterrows()])
        cc.index.set_names('assigned', level=2, inplace=True)
        resres.append(cc)

    resres_df = pd.concat(resres, axis=0)
    datasets_columns = [dataset_id_to_mnemonic[ds] for ds in d]
    resres_df.columns = datasets_columns
    resres_df = resres_df[resres_df.sum(axis=1) > 0]
    if len(datasets_columns) == 2:
        resres_df[f'Diff'] = resres_df.apply(lambda row: row[datasets_columns[0]] - row[datasets_columns[1]], axis=1)
        resres_df[f'Ratio'] = resres_df.apply(lambda row: row[datasets_columns[0]] / row[datasets_columns[1]], axis=1)
    resres_df['variance^(1/2)'] = resres_df.apply(lambda row: np.var(row) ** 0.5, axis=1)
    return resres_df


@st.cache(allow_output_mutation=True, show_spinner=False)
def select_datapoints(dataset_mnemonic, indices, batch, limit=50):
    dataset = get_dataset(dataset_mnemonic_to_id[dataset_mnemonic], batch)
    res = {}
    for i, datapoint in enumerate(dataset):
        i += batch * CHUNK_SIZE
        if i in indices:
            res[i] = datapoint
        if len(res) >= limit:
            break
    return res, len(dataset) == CHUNK_SIZE


@st.cache(show_spinner=False)
def compute_metric(model, dataset, metric) -> float:
        df = read_labeled_dataset(model, dataset)
        if metric == 'accuracy':
            return accuracy(df)
        elif metric == 'precision':
            return precision(df)
        elif metric == 'f1':
            return f1(df)
        elif metric == 'recall':
            return recall(df)
        else:
            raise ValueError(f'Unknown metric: {metric}')


def accuracy(df):
    total = len(df)
    correct = TP(df) + TN(df)
    return float(correct) / total


def precision(df):
    predicted_positive = (TP(df) + FP(df))
    if predicted_positive == 0:
        return float("nan")
    return float(TP(df)) / predicted_positive


def recall(df):
    actual_positives = (TP(df) + FN(df))
    if actual_positives == 0:
        return float("nan")
    return float(TP(df)) / actual_positives


def f1(df):
    prec = precision(df)
    rec = recall(df)
    return 2 * prec * rec / (prec + rec)


def TP(df):
    return len(df[(df['label'] == 1) & (df['prob_CommitLabel.BugFix'] > 0.5)])


def TN(df):
    return len(df[(df['label'] == 0) & (df['prob_CommitLabel.BugFix'] < 0.5)])


def FP(df):
    return len(df[(df['label'] == 0) & (df['prob_CommitLabel.BugFix'] > 0.5)])


def FN(df):
    return len(df[(df['label'] == 1) & (df['prob_CommitLabel.BugFix'] < 0.5)])
