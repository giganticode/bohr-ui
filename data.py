import json
import re
import subprocess
from typing import List, Dict, Optional, NewType, Any

import jsonlines as jsonlines
import streamlit as st

import pandas as pd
import numpy as np
import dvc.api
from bohrruntime.core import load_workspace
from dvc.exceptions import PathMissingError
from sklearn.metrics import f1_score

from tqdm import tqdm

from config import get_mnemonic_for_dataset
from vcs import get_path_to_revision

bohr_bugginess_repo = 'https://github.com/giganticode/bohr-workdir-bugginess'
diff_classifier_repo = 'https://github.com/giganticode/diff-classifier'
bohr_repo = 'https://github.com/giganticode/bohr'

DATASET_DEBUGGING_EXPERIMENT = 'dataset_debugging'

CHUNK_SIZE = 10000
TRANSFORMER_REGEX = re.compile('fine_grained_changes_transformer_(\\d+)')
KEYWORD_REGEX = re.compile('bug.*_message_keyword_(.*)')

ModelMetadata = NewType('ModelMetadata', Dict[str, Any])


@st.cache(show_spinner=False)
def get_label_matrix_locally(tmp_path):
    with st.spinner(f'Loading label matrix from {tmp_path}'):
        return pd.read_pickle(tmp_path)


@st.cache(show_spinner=False)
def load_used_bohr_commit_sha() -> str:
    path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', True)
    workspace = load_workspace(path_to_revision)
    for exp in workspace.experiments:
        if exp.name == DATASET_DEBUGGING_EXPERIMENT:
            return exp.revision
    raise AssertionError(f'Experiment {DATASET_DEBUGGING_EXPERIMENT} not found.')


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_label_matrix(dataset_name: str) -> pd.DataFrame:
    path = f'runs/bugginess/{DATASET_DEBUGGING_EXPERIMENT}/{dataset_name}/heuristic_matrix.pkl'
    path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', True)
    with st.spinner(f'Loading label matrix for dataset `{dataset_name}`'):
        with st.spinner(f'Reading `{path}` from `{bohr_bugginess_repo}`'):
            subprocess.run(["dvc", "pull", path], cwd=path_to_revision)
    return get_label_matrix_locally(f'{path_to_revision}/{path}')


@st.cache(allow_output_mutation=True, show_spinner=False)
def read_labeled_dataset(model_metadata: ModelMetadata, selected_dataset_name: str, indices):
    model_name = model_metadata["name"]
    model_type = model_metadata['model']
    with st.spinner(f'Loading labels by model `{model_name}` for dataset `{selected_dataset_name}`'):
        if model_type == 'label model':
            df = read_labeled_dataset_from_bohr(model_name, selected_dataset_name)
        elif model_type == 'transformer':
            df = read_labeled_dataset_from_transformers(model_name, selected_dataset_name)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
    return df.iloc[indices, :] if indices is not None else df


@st.cache(show_spinner=False)
def get_fired_indexes(dataset, heuristic, value):
    label_matrix = get_label_matrix(dataset)
    h_values = label_matrix[heuristic]
    h_values = h_values[h_values == value]
    return h_values.index.values.tolist()


@st.cache(show_spinner=False)
def load_transformer_metadata() -> Dict[str, ModelMetadata]:
    path_to_revision = get_path_to_revision(diff_classifier_repo, 'master', True)
    full_path = path_to_revision / 'models'
    res = {}
    with st.spinner(f'Reading models metadata from `{bohr_bugginess_repo}`'):
        for model_dir in full_path.iterdir():
            path = model_dir / 'task.metadata'
            if not path.exists():
                continue
            with open(path) as f:
                dct = json.load(f)
                res[dct['name']] = dct
    return res


class DatasetNotFound(Exception):
    def __init__(self, repo, path):
        self.repo = repo
        self.path = path


@st.cache(show_spinner=False)
def read_labeled_dataset_from_transformers(model_name: str, selected_dataset):
    label_to_int = {'NonBugFix': 0, 'BugFix': 1}
    path = f'models/{model_name}/assigned_labels_{selected_dataset}.csv'
    try:
        with st.spinner(f'Reading `{path}` from `{diff_classifier_repo}`'):
            with dvc.api.open(path, repo=diff_classifier_repo) as f:
                df = pd.read_csv(f).rename(columns={'true_label': 'label'}, inplace=False)
    except PathMissingError as ex:
        raise DatasetNotFound(diff_classifier_repo, path) from ex
    if 'label' in df.columns:
        df.loc[:, 'label'] = df.apply(lambda row: label_to_int[row["label"]], axis=1)
    else:
        #df.loc[:, 'label'] = 1
        raise AssertionError()
    df.loc[:, 'prob_CommitLabel.BugFix'] = df.apply(
        lambda row: (row['probability'] if row['prediction'] == 'BugFix' else (1 - row['probability'])), axis=1)
    return df


@st.cache(show_spinner=False)
def read_labeled_dataset_from_bohr(model_name: str, selected_dataset_name: str):
    label_to_int = {'CommitLabel.NonBugFix': 0, 'CommitLabel.BugFix': 1}
    path = f'runs/bugginess/{model_name}/{selected_dataset_name}/labeled.csv'
    path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', True)
    subprocess.run(["dvc", "pull", path], cwd=path_to_revision)
    full_path = f'{path_to_revision}/{path}'
    with st.spinner(f'Reading `{full_path}` from `{bohr_bugginess_repo}`'):
        print(f'Reading labeled dataset from bohr repo at: {full_path}')
        with open(full_path) as f:
            df = pd.read_csv(f)
    if 'label' in df.columns:
        df.loc[:, 'label'] = df.apply(lambda row: label_to_int[row["label"]], axis=1)
    return df


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_dataset_chunk(dataset, chunk: int) -> List[Dict]:
    path = f'cached-datasets/{dataset}.jsonl'
    with st.spinner(f"Loading dataset `{path}` from `{bohr_bugginess_repo}`"):
        path_to_revision = get_path_to_revision(bohr_bugginess_repo, 'master', True)
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


def decompose_heuristic_name(s):
    matcher = KEYWORD_REGEX.fullmatch(s)
    if matcher:
        return 'keyword', matcher.group(1)
    else:
        matcher = TRANSFORMER_REGEX.fullmatch(s)
        if matcher:
            return 'transformer', f'{int(matcher.group(1)) + 10}>p>{matcher.group(1)}'
        else:
            return 'file metric', s


@st.cache(show_spinner=False)
def get_lfs(dataset_id) -> List[str]:
    with st.spinner("Loading labeling function list ..."):
        label_matrix = get_label_matrix(dataset_id)
        return list(label_matrix.columns)


@st.cache
def compute_lf_coverages(dataset_ids, indexes: Optional[Dict[str, List[int]]] = None):
    lf_values = [0, 1]
    all_datasets_coverage = []
    for lf_value in lf_values:
        all_datasets_one_value = []
        for dataset in dataset_ids:
            label_matrix = get_label_matrix(dataset)
            if indexes is not None:
                label_matrix = label_matrix.iloc[indexes[dataset], :]
            ln = len(label_matrix)
            coverage_by_value = ((label_matrix == lf_value).sum(axis=0) / float(ln)) if ln > 0 else (label_matrix == lf_value).sum(axis=0)
            all_datasets_one_value.append(coverage_by_value)
        all_datasets_one_value_df = pd.concat(all_datasets_one_value, axis=1)
        all_datasets_one_value_df.index = pd.MultiIndex.from_tuples([(*decompose_heuristic_name(k), lf_value) for k, v in all_datasets_one_value_df.iterrows()], names=['LF type', 'details', 'value by LF'])
        all_datasets_coverage.append(all_datasets_one_value_df)

    all_datasets_coverage_df = pd.concat(all_datasets_coverage, axis=0)
    datasets_columns = [get_mnemonic_for_dataset(ds) for ds in dataset_ids]
    all_datasets_coverage_df.columns = datasets_columns
    all_datasets_coverage_df = all_datasets_coverage_df[all_datasets_coverage_df.sum(axis=1) > 0]
    if not all_datasets_coverage_df.empty:
        if len(datasets_columns) == 2:
            all_datasets_coverage_df[f'diff'] = all_datasets_coverage_df.apply(lambda row: row[datasets_columns[0]] - row[datasets_columns[1]], axis=1)
            all_datasets_coverage_df[f'ratio'] = all_datasets_coverage_df.apply(lambda row: row[datasets_columns[0]] / row[datasets_columns[1]], axis=1)
        all_datasets_coverage_df['variance^(1/2)'] = all_datasets_coverage_df.apply(lambda row: np.var(row) ** 0.5, axis=1)
    return all_datasets_coverage_df


class TrueLabelNotFound(Exception):
    pass


@st.cache(show_spinner=False)
def compute_metric(df, metric, randomize_abstains) -> float:
    if randomize_abstains and metric != 'certainty':
        predicted_continuous = df.apply(lambda row: (np.random.random() if np.isclose(row['prob_CommitLabel.BugFix'], 0.5) else row['prob_CommitLabel.BugFix']), axis=1)
    else:
        predicted_continuous = df['prob_CommitLabel.BugFix']
    predicted = (predicted_continuous > 0.5).astype(int)
    if 'label' not in df.columns:
        raise TrueLabelNotFound()
    actual = df['label']
    if metric == 'certainty':
        return certainty(predicted_continuous)
    elif metric == 'accuracy':
        return accuracy(predicted, actual)
    elif metric == 'precision':
        return precision(predicted, actual)
    elif metric == 'f1 (macro)':
        return f1(predicted, actual)
    elif metric == 'recall':
        return recall(predicted, actual)
    else:
        raise ValueError(f'Unknown metric: {metric}')


def certainty(predicted_continuous):
    """
    >>> certainty([0.85, 1, 1])
    0.90
    """
    return ((predicted_continuous - 0.5) * 2).abs().mean()
    

def accuracy(predicted, actual):
    total = len(predicted)
    correct = TP(predicted, actual) + TN(predicted, actual)
    return float(correct) / total


def precision(predicted, actual):
    predicted_positive = (TP(predicted, actual) + FP(predicted, actual))
    if predicted_positive == 0:
        return float("nan")
    return float(TP(predicted, actual)) / predicted_positive


def recall(predicted, actual):
    actual_positives = (TP(predicted, actual) + FN(predicted, actual))
    if actual_positives == 0:
        return float("nan")
    return float(TP(predicted, actual)) / actual_positives


def f1(predicted, actual):
    # prec = precision(predicted, actual)
    # rec = recall(predicted, actual)
    # return 2 * prec * rec / (prec + rec)
    return f1_score(actual, predicted, average='macro')


def TP(predicted, actual):
    """
    >>> TP([1, 0, 0], [1, 0, 1])
    1
    """
    return sum(a == 1 and p == 1 for p, a in zip(predicted, actual))


def TN(predicted, actual):
    """
    >>> TP([1, 0, 0], [1, 0, 1])
    1
    """
    return sum(a == 0 and p == 0 for p, a in zip(predicted, actual))


def FP(predicted, actual):
    """
    >>> TP([1, 0, 0], [1, 0, 1])
    0
    """
    return sum(a == 0 and p == 1 for p, a in zip(predicted, actual))


def FN(predicted, actual):
    """
    >>> TP([1, 0, 0], [1, 0, 1])
    1
    """
    return sum(a == 1 and p == 0 for p, a in zip(predicted, actual))
