import random
from typing import Optional, Tuple, List, Dict, Union, Iterable

import pandas as pd
import streamlit as st

import numpy as np
import seaborn as sns
from bohrapi.artifacts import Commit
from bohrruntime.heuristics import load_heuristic_by_name
from bohrruntime.util.paths import AbsolutePath
from dvc.exceptions import PathMissingError
from matplotlib import pyplot as plt
from pandas import MultiIndex
from sklearn.metrics import confusion_matrix

from vcs import get_path_to_revision

st.set_page_config(
    page_title="Bugginess",
    layout="wide",
)

from config import datasets_with_labels, datasets_without_labels, get_mnemonic_for_dataset, label_models_metadata
from data import read_labeled_dataset, get_dataset, compute_lf_coverages, compute_metric, get_fired_indexes, get_lfs, \
    bohr_repo, load_used_bohr_commit_sha, load_transformer_metadata, DatasetNotFound, TrueLabelNotFound, \
    NoDatapointsFound, SubsetSelectionCriterion, LFResult, get_weights, get_fired_heuristics, get_label_matrix, \
    get_all_dataset_fired_values, ModelMetadata

cm = sns.light_palette("green", as_cmap=True)


def get_possible_lf_values(lf_name: str, dataset_names: str) -> List[int]:
    all_possible_values = set()
    for dataset_name in dataset_names:
        try:
            label_matrix = get_label_matrix(dataset_name)
            all_possible_values.update(label_matrix[lf_name].tolist())
        except FileNotFoundError:
            pass
    all_possible_values.remove(-1)
    return list(all_possible_values)


def show_lf_selection(col2, col3, col4, prefix, default_indices, dataset_names) -> Tuple[str, Optional[int]]:
    all_dataset_fired_values = get_all_dataset_fired_values(dataset_names)
    fired_lfs = [name for name, s in all_dataset_fired_values.items() if len(s) > 0]
    lf = col2.selectbox('to which LF:', fired_lfs, key=f'{prefix}.heuristic', index=default_indices[0] if default_indices else 0)
    possible_values = all_dataset_fired_values[lf]
    if len(possible_values) > 1:
        value = col3.selectbox('assigned value:', possible_values, key=f'{prefix}.value', index=default_indices[1] if default_indices else 0)
    elif len(possible_values) == 1:
        value = possible_values[0]
    else:
        raise ValueError('Selected LF does not fire on these datasets.')
    commit = load_used_bohr_commit_sha()
    path_to_revision = get_path_to_revision(bohr_repo, 'bohr-0.5', True, commit)
    _, path = load_heuristic_by_name(lf, Commit, AbsolutePath(path_to_revision) / 'heuristics', return_path=True)
    github_link = f'https://github.com/giganticode/bohr/tree/{commit}/' + str(path)
    col4.write('')
    col4.write('')
    col4.write(f'`View labeling function on `[GitHub]({github_link})')
    return lf, value


def datapoints_subset_ui(dataset_names, prefix, default_indices: Optional[Tuple[int, int]] = None) -> SubsetSelectionCriterion:
    col1, col2, col3, col4 = st.columns(4)
    col1.checkbox('Subset of data points', value=False, key=f'{prefix}.dataset_subset',
                  help='Limit to those data points on which a specific LF was fired.\n\n'
                       'The list also contains slicing function that are technically LFs but serve '
                       'the purpose of selecting important subsets of data rather assigning labels to them,'
                       ' e.g. `small change`, `large change`')
    subset_selection_criterion = SubsetSelectionCriterion()
    show_next_lf_ui = st.session_state[f'{prefix}.dataset_subset']
    counter = 0
    while show_next_lf_ui:
        lf, value = show_lf_selection(col2, col3, col4, f'{prefix}.{counter}', default_indices, dataset_names)
        subset_selection_criterion.lf_results.append(LFResult(lf, value))
        _, col2, col3, col4 = st.columns(4)
        col2.checkbox('+ Add LF to filter', value=False, key=f"{prefix}.rrr.{counter}")
        show_next_lf_ui = st.session_state[f"{prefix}.rrr.{counter}"]
        counter += 1
        _, col2, col3, col4 = st.columns(4)

    return subset_selection_criterion


def choose_single_dataset_ui(datasets, prefix, default_indices = None) -> Tuple[str, SubsetSelectionCriterion]:
    chosen_dataset = st.selectbox('Select dataset to be analyzed:', datasets, key=f'{prefix}.msl', index=default_indices[0] if default_indices else 0, format_func=get_mnemonic_for_dataset)
    subset_selection_criterion = None
    if chosen_dataset is not None:
        subset_selection_criterion = datapoints_subset_ui(chosen_dataset, prefix, (default_indices[1], default_indices[2]))
    return chosen_dataset, subset_selection_criterion


def choose_datasets_ui(datasets, prefix) -> Tuple[List[str], SubsetSelectionCriterion]:
    chosen_datasets = st.multiselect('Select dataset(s) to be analyzed:', options=datasets,
                                     default=[datasets[5], datasets[0], datasets[6]], key=f'{prefix}.msl', format_func=get_mnemonic_for_dataset)
    subset_selection_criterion = None
    if len(chosen_datasets) > 0:
        subset_selection_criterion = datapoints_subset_ui(chosen_datasets, prefix)
    return chosen_datasets, subset_selection_criterion


def display_coverage() -> None:
    try:
        chosen_datasets, subset_selection_criterion = choose_datasets_ui(sorted(datasets_with_labels.keys()) + datasets_without_labels, 'lf_coverage')
        indices = None
        if subset_selection_criterion:
            indices = {dataset: get_fired_indexes(dataset, subset_selection_criterion) for dataset in chosen_datasets}
        if len(chosen_datasets) > 0:
            st_placeholder = st.empty()
            options = ['transformer', 'keyword', 'file metric']
            cols = st.columns(len(options))
            for col, option in zip(cols, options):
                col.checkbox(option, value=True, key=option)
            coverages_df = compute_lf_coverages([mn for mn in chosen_datasets], indices)
            lfs = coverages_df.columns
            lfs_to_include = [option for option in options if st.session_state[option]]
            if len(lfs_to_include) == 0:
                st.warning('You filtered out all the LFs!')
                return
            filtered_coverages_df = coverages_df[coverages_df.index.isin(lfs_to_include, level=0)]
            filtered_coverages_df.reset_index(inplace=True)
            df_styler = filtered_coverages_df.style.format({key: "{:.2%}" for key in lfs})
            df_styler = df_styler.background_gradient(cmap=cm)
            if not filtered_coverages_df.empty:
                st_placeholder.write(df_styler)
                st.write('`Labeling functions that have zero coverage over all the datasets are omited`')
            else:
                if len(subset_selection_criterion.lf_results) == 1:
                    lf_result = subset_selection_criterion.lf_results[0]
                    st_placeholder.info(f'There are no fired data points in any of the datasets. '
                                        f'Are you sure that LF `{lf_result.lf}` can assign value `{lf_result.value}`?')
                else:
                    st_placeholder.info(f'There are no fired data points in any of the datasets. Try simplifying the query.')
            st.download_button('Download', filtered_coverages_df.to_csv(), file_name='lf_values.csv')
    except PathMissingError as ex:
        st.warning("Selected dataset is not found. Was it uploaded to dvc remote?")
        st.exception(ex)


def display_label_model_filter_ui(all_label_models: List[ModelMetadata]) -> List[ModelMetadata]:

    st.checkbox('Include label models', value=False, key='include_label_models')
    if st.session_state['include_label_models']:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.radio('Labeling functions', ['zeros', 'random', 'keywords', 'all heuristics', 'gitcproc', 'gitcproc_orig', 'all'],
                   key="labeling_functions", index=4,
                   help='keywords: only keyword heuristics\n\n'
                        'all heuristics: keywords + file metrics + transformer trained on conventional commits\n\n'
                        'gitcproc:  9 "bugfix" keyword heuristics, "non-bugfix" if none matched\n\n'
                        'gitcproc_orig: 9 "bugfix" keyword heuristics, abstain if none matched')
        lfs_radio = st.session_state['labeling_functions']
        if lfs_radio != 'all':
            all_label_models = [model for model in all_label_models if (model['label_source'] == lfs_radio)]
        col2.radio('Issues', ['with issues', 'without issues', 'issue labels', 'all'], key="issues", index=2)
        issues_radio = st.session_state['issues']
        if issues_radio != 'all':
            all_label_models = [model for model in all_label_models if (model['issues'] == issues_radio)]
        col3.radio('Train set', ['commits_200k_files', 'commits_200k_files_no_merges', 'all'], key="train_set", index=2)
        train_set = st.session_state['train_set']
        if train_set != 'all':
            all_label_models = [model for model in all_label_models if (model['train_dataset'] == train_set)]
    else:
        all_label_models = []
    return all_label_models


def display_transformer_filter_ui(all_transformer_models: Iterable[ModelMetadata]) -> List[ModelMetadata]:

    st.checkbox('Include transformer models', value=True, key='include_transformer_models')
    if st.session_state['include_transformer_models']:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.radio('Labels from label model trained on lfs:', ['keywords', 'all heuristics', 'gitcproc', 'gitcproc_orig', 'all'],
                   key="label_source", index=4,
                   help='keywords: only keyword heuristics\n\n'
                        'all heuristics: keywords + file metrics + transformer trained on conventional commits\n\n'
                        'gitcproc:  9 "bugfix" keyword heuristics, "non-bugfix" if none matched\n\n'
                        'gitcproc_orig: 9 "bugfix" keyword heuristics, abstain if none matched')
        label_source = st.session_state.label_source
        if label_source != 'all':
            all_transformer_models = [model for model in all_transformer_models if (model['label_source'] == label_source)]

        col2.radio('Issues', ['with issues', 'without issues', 'all'], key="issues1", index=2)
        issues = st.session_state.issues1
        if issues != 'all':
            all_transformer_models = [model for model in all_transformer_models if (model['issues'] == issues)]

        col3.radio('Train set', ['commits_200k_files', 'commits_200k_files_no_merges', 'all'], key="train_set1", index=2)
        train_set = st.session_state['train_set1']
        if train_set != 'all':
            all_transformer_models = [model for model in all_transformer_models if (model['train_dataset'] == train_set)]

        col4.radio('Transformer trained on', ['only message', 'only change', 'message and change', 'all'], key="input_data", index=3)
        input_data = st.session_state['input_data']
        if input_data != 'all':
            all_transformer_models = [model for model in all_transformer_models if (model['trained_on'] == input_data)]
        col5.checkbox('Include soft labels', key='soft_labels', value=True)
        if not st.session_state['soft_labels']:
            all_transformer_models = [model for model in all_transformer_models if not model['soft_labels']]
    else:
        all_transformer_models = []
    return all_transformer_models


def collect_models(models: List[ModelMetadata], dataset_name: str, indices: List[int]):
    n_exp = 0
    for i, model_metadata in enumerate(models):
        model_name = model_metadata['name']
        try:
            raw_df = read_labeled_dataset(model_metadata, dataset_name, indices if indices is not None else None)
        except DatasetNotFound as ex:
            st.warning(f'Dataset was not found: {ex}, skipping model: {model_metadata["name"]}')
            continue
        n_exp += 1
        if raw_df.empty:
            raise NoDatapointsFound()
        if i == 0:
            if 'label' in raw_df.columns:
                prep_df: pd.DataFrame = raw_df[['sha', 'message', 'label']]
                prep_df.rename(columns={'label': 'true label'}, inplace=True)
            else:
                prep_df: pd.DataFrame = raw_df[['sha', 'message']]
        if 'ind_perf_metrics' not in st.session_state:
            st.session_state['ind_perf_metrics'] = 'probability'
        if st.session_state['ind_perf_metrics'] == 'accuracy':
            if 'label' in raw_df.columns:
                prep_df.loc[:, model_name] = 1 - (raw_df["prob_BugFix"] - raw_df["label"]).abs()
            else:
                raise TrueLabelNotFound('Cannot calc accuracy for dataset without ground truth labels')
        elif st.session_state['ind_perf_metrics'] == 'probability':
            prep_df.loc[:, model_name] = raw_df["prob_BugFix"]
        else:
            raise ValueError(f'Unknown value {st.session_state.ind_perf_metrics}')
    return prep_df, n_exp



def display_model_perf_on_ind_data_points(models: List[ModelMetadata], dataset_name: str,
                                          indices: List[int], subset_selection_criterion: SubsetSelectionCriterion):
    try:
        prep_df, n_exp = collect_models(models, dataset_name, indices)
        model_names = list(map(lambda m: m['name'], models))
        chosen_metric = st.session_state['ind_perf_metrics']
        if chosen_metric == 'accuracy':
            prep_df.loc[:, 'how_often_precise'] = prep_df[model_names].apply(lambda row: (row > 0.5).sum() / float(n_exp), axis=1)
            sort_columns=['how_often_precise', 'variance']
        elif chosen_metric == 'probability':
            if 'true labels' in prep_df:
                precision = 1 - (prep_df[model_names].sub(prep_df["true label"], axis=0)).abs()
                prep_df.loc[:, 'how_often_precise'] = precision.apply(lambda row: (row > 0.5).sum() / float(n_exp), axis=1)
            sort_columns=['variance']
        else:
            raise ValueError(f'Unknown metric: {st.session_state["ind_perf_metrics"]}')
        prep_df.loc[:, 'variance'] = prep_df[model_names].apply(lambda row: np.var(row), axis=1)

        prep_df.sort_values(sort_columns, inplace=True, ascending=True)

        df_styler = prep_df.style.format({key: "{:.2%}" for key in list(model_names) + sort_columns})
        df_styler = df_styler.background_gradient(cmap=cm)

        desc = format_subset_description(dataset_name, subset_selection_criterion, len(prep_df))
        st.write(desc)
        st.write(df_styler)
        st.download_button('Download', data=prep_df.to_csv(), file_name='preformance_data_points.csv')
    except TrueLabelNotFound:
        st.warning('Cannot compute accuracy for selected dataset: true labels not found')
    except NoDatapointsFound:
        st.warning('No datapoints')
    st.radio('', ['accuracy', 'probability'], key='ind_perf_metrics', format_func=format_indiv_radio)


def format_subset_description(dataset_name: str, subset_selection_criterion: SubsetSelectionCriterion, n_datapoints: int):
    desc = f'`Displaying data points from dataset {dataset_name}'
    desc += str(subset_selection_criterion)
    desc += f', n datapoints: {n_datapoints}`'
    return desc


def format_indiv_radio(s):
    map = {
        'accuracy': 'Accuracy on datapoint (prob assigned to true label)',
        'probability': 'Probability assigned to BugFix class',
    }
    return map[s]


def convert_metrics_to_relative(df):
    rel_matrix = np.zeros_like(df)
    for i in range(rel_matrix.shape[0] - 1):
        for j in range(rel_matrix.shape[1]):
            if st.session_state.comp_mode == 'compare to previous model':
                rel_matrix[i+1, j] = df.iloc[i+1,j] - df.iloc[i, j]
            elif st.session_state.comp_mode == 'compare to baseline':
                rel_matrix[i+1, j] = df.iloc[i+1, j] - df.iloc[0, j]
            else:
                raise ValueError(f'{st.session_state.comp_mode}')
    return pd.DataFrame(rel_matrix, index=df.index, columns=df.columns)


def format_normalization_labels(s):
    map = {
        'None' : 'None',
        'true': 'normalize over true labels',
        'pred': 'normalize over preticted labels',
        'all': 'normalize by total number of samples',
    }
    return map[s]


def display_confusion_matrix(label_models: List[ModelMetadata], transformers: List[ModelMetadata], selected_datasets, indices, subset_selection_criterion: SubsetSelectionCriterion):
    st.radio('Normalization', options=['None', 'true', 'pred', 'all'], key='normalize', format_func=format_normalization_labels)
    normalize = st.session_state['normalize']
    if normalize == 'None':
        normalize = None
    for dataset in selected_datasets:
        dataset_description = f'{get_mnemonic_for_dataset(dataset)}'
        dataset_description += str(subset_selection_criterion)
        st.write(f"#### {dataset_description}")
        layout = get_layout_for_confusion(len(label_models) + len(transformers))
        cols = [c for row in layout for c in st.columns(row)]
        for i, model in enumerate(label_models + transformers ):
            confusion(dataset, model, indices, normalize, cols[i])


def create_metrics_dataframe(label_models: List[ModelMetadata], transformers: List[ModelMetadata], selected_datasets: List[str], indices) -> pd.DataFrame:
    models = label_models + transformers
    matrix = []
    excluded_models = []
    zero_datapoints_for_some_datasets = False
    for model_metadata in models:
        row = []
        st.spinner(f'Computing metrics for model `{model_metadata["name"]}`')
        try:
            for dataset_name in selected_datasets:
                st.spinner(f'Computing metrics for dataset `{dataset_name}`')
                abstains_present = model_metadata['model'] == 'label model'
                df = read_labeled_dataset(model_metadata, dataset_name, indices[dataset_name] if indices is not None else None)
                if not df.empty:
                    res = compute_metric(df, st.session_state.metric, abstains_present)
                else:
                    res = np.nan
                    zero_datapoints_for_some_datasets = True
                row.append(res)
            matrix.append(row)
        except DatasetNotFound as ex:
            st.warning(f'Dataset was not found: {ex}, skipping model: {model_metadata["name"]}')
            excluded_models.append(model_metadata)
    if zero_datapoints_for_some_datasets:
        st.warning('Zero matched data points for some datasets.')

    label_model_index_tuples = [(
        model['name'],
        model['model'],
        model['label_source'],
        model['issues'],
        model['train_dataset'],
        'N/A',
        'N/A',
        'N/A',
    ) for model in label_models if model not in excluded_models]

    transformer_index_tuples = [(
        model['name'],
        model['model'],
        model['label_source'],
        model['issues'],
        model['train_dataset'],
        model['trained_on'],
        'yes' if model['soft_labels'] == 1 else 'no',
        model['augmentation'],
    ) for model in transformers if model not in excluded_models]
    ind = MultiIndex.from_tuples(label_model_index_tuples + transformer_index_tuples, names=['id', 'model', 'label source', 'issues', 'trained on', 'dataset type', 'soft labels', 'data aug'])
    metrics_dataframe = pd.DataFrame(matrix, index=ind, columns=selected_datasets)
    metrics_dataframe = metrics_dataframe.sort_values(by=selected_datasets[0], inplace=False)
    return metrics_dataframe


def display_model_metrics(label_models: List[ModelMetadata], transformers: List[ModelMetadata], selected_datasets: List[str], indices, subset_selection_criterion: SubsetSelectionCriterion) -> List[ModelMetadata]:
    if 'metric' not in st.session_state:
        st.session_state['metric'] = 'accuracy'
    if 'rel_imp' not in st.session_state:
        st.session_state['rel_imp'] = False
    if 'comp_mode' not in st.session_state:
        st.session_state['comp_mode'] = 'compare to baseline'

    metrics_dataframe = create_metrics_dataframe(label_models, transformers, selected_datasets, indices)
    if st.session_state.rel_imp:
        metrics_dataframe = convert_metrics_to_relative(metrics_dataframe)
    metrics_dataframe.reset_index(level=[1, 2, 3, 4, 5, 6, 7], inplace=True)
    metrics_styler = metrics_dataframe.style.format({key: "{:.2%}" for key in selected_datasets})
    if st.session_state.rel_imp:
        metrics_styler = metrics_styler.background_gradient(cmap=sns.blend_palette("rg", as_cmap=True), vmin=-0.3, vmax=0.2, )
    else:
        metrics_styler = metrics_styler.background_gradient(cmap=cm)
    st.write(metrics_styler)
    st.download_button('Download', data=metrics_dataframe.to_csv(), file_name='metrics.csv')
    c1, c2 = st.columns(2)
    c1.radio('Metric', ['accuracy', 'f1 (macro)', 'precision', 'recall', 'certainty'],
             key='metric', help='')
    c2.checkbox(label="Show relative improvement", value=False, key='rel_imp')
    if st.session_state.rel_imp:
        c2.radio('', ['compare to baseline', 'compare to previous model'], key='comp_mode')
    st.checkbox('Show confusion matrix(es)', value=False, key='conf_matrix_checkbox')
    if st.session_state['conf_matrix_checkbox']:
        display_confusion_matrix(label_models, transformers, selected_datasets, indices, subset_selection_criterion)
    return [model_metadata for model_metadata in label_models + transformers if model_metadata['name'] in list(metrics_dataframe.index)]


def display_datapoint_search_ui(dataset, dataset_name, filtered_models):
    st.text_input("Get full information about data point ...", key='rt', placeholder='Start typing a commit hash')
    if st.session_state.rt != '':
        datapoint, seq = get_datapoint_by_id_beginning(dataset, st.session_state.rt, return_seq=True)
        if st.checkbox('Show weights of all LFs fired on this data point (for LMs only)', value=False, key='all_sp_weights',
                       help='N/A means that the heuristic was fired on this data point but is not used by current model'):
            subset_criterion_criterion = get_fired_heuristics(dataset_name, seq)
            show_model_weights(filtered_models, subset_criterion_criterion)
        if datapoint:
            st.write(f'https://github.com/{datapoint["owner"]}/{datapoint["repo"]}/commit/{datapoint["_id"]}')
            st.write(datapoint)
        else:
            st.warning('Data point not found.')


def get_datapoint_by_id_beginning(dataset, id: str, return_seq: bool = False) -> Union[Optional[Dict],
                                                                                  Tuple[Optional[Dict], Optional[int]]]:
    for i, datapoint in enumerate(dataset):
        if datapoint['_id'].startswith(id):
            return (datapoint, i) if return_seq else datapoint
    return (None, None) if return_seq else None


def get_layout_for_confusion(n):
    map = {
        1: [4],
        2: [4],
        3: [4],
        4: [4],
        5: [3, 3],
        6: [4, 4],
        7: [4, 4],
        8: [4, 4],
        9: [3, 3, 3],
        10: [4, 4, 4],
        11: [4, 4, 4],
        12: [4, 4, 4],
        13: [4, 4, 5],
        14: [4, 4, 4, 4],
        15: [4, 4, 4, 4],
        16: [4, 4, 4, 4],
        17: [4, 4, 4, 4, 5],
        18: [4, 4, 4, 5, 5],

    }
    return map[n]


def plot_confusion_matrix(y_true, y_pred,
                          normalize,
                          title='',
                          cmap="viridis", ax=None):

    # Compute confusion matrix
    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
    cm = np.nan_to_num(cm)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           # xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    return ax


def confusion(dataset_name: str, model_metadata: ModelMetadata, indices, normalize, handle):
    try:
        labeled_dataset = read_labeled_dataset(model_metadata, dataset_name, indices[dataset_name] if indices is not None else None)
    except DatasetNotFound:
        handle.write(f'Model {model_metadata["name"]} is not found.')
        return
    if labeled_dataset.empty:
        handle.write('No datapoints.')
        return
    predicted_continuous = labeled_dataset['prob_CommitLabel.BugFix']
    predicted = (predicted_continuous > 0.5).astype(int)
    ax= plt.subplot()
    title_details = "\n".join(map(lambda m: str(m), model_metadata.values()))
    title = f'{model_metadata["name"]}\n\n{title_details}'
    disp = plot_confusion_matrix(labeled_dataset['label'], predicted, normalize=normalize, title=title, ax=ax)

    handle.pyplot(disp.figure)


def show_model_weights(filtered_models, subset_selection_criterion: SubsetSelectionCriterion):
    ln = len(filtered_models)
    if ln <= 0:
        st.warning('Please select at leats one model.')
    elif ln > 3:
        st.warning('Can display weight for at most 3 models.')
    else:
        weights_list = get_weights(filtered_models, subset_selection_criterion)
        columns = st.columns(ln)
        for c, wl, model in zip(columns, weights_list, filtered_models):
            title = f'#### {model["name"]}\n\n{model["label_source"]} - {model["issues"]}'
            c.write(title)
            c.write(wl)


def display_bugginess_performance():
    transformer_metadata = load_transformer_metadata()
    filtered_label_models = display_label_model_filter_ui(label_models_metadata)
    filtered_transformers = display_transformer_filter_ui(list(transformer_metadata.values()))
    filtered_models = filtered_label_models + filtered_transformers
    selected_datasets, subset_criterion_criterion = choose_datasets_ui(sorted(datasets_with_labels.keys()), 'model_perf')
    indices = None
    if subset_criterion_criterion:
        indices = {dataset: get_fired_indexes(dataset, subset_criterion_criterion) for dataset in selected_datasets}
    if len(filtered_models) == 0:
        st.warning('No model satisfying chosen conditions found.')
    elif len(selected_datasets) == 0:
        st.warning('No datasets selected.')
    else:
        filtered_models = display_model_metrics(filtered_label_models, filtered_transformers, selected_datasets, indices, subset_criterion_criterion)
    return filtered_models


def display_individual_data_point_debugging(filtered_models):
    default_indices = (
        sorted(datasets_with_labels.keys()).index(st.session_state[f'model_perf.msl'][0]) if (f'model_perf.msl' in st.session_state and st.session_state[f'model_perf.msl']) else 0,
        0,
        st.session_state[f'model_perf.value'] if f'model_perf.value' in st.session_state else 0,
    )
    selected_dataset, subset_criterion_criterion = choose_single_dataset_ui(sorted(datasets_with_labels.keys()) + datasets_without_labels, 'model_perf_ind', default_indices)
    if st.checkbox('Show weights assigned to selected LFs (for LMs only)', value=False, key='all_sp_weights_ind'):
        show_model_weights(filtered_models, subset_criterion_criterion)
    indices = None
    if subset_criterion_criterion:
        indices = get_fired_indexes(selected_dataset, subset_criterion_criterion)
    if selected_dataset is not None and len(filtered_models) > 0:
        display_model_perf_on_ind_data_points(filtered_models, selected_dataset, indices, subset_criterion_criterion)

        dataset = get_dataset(selected_dataset)
        display_datapoint_search_ui(dataset, selected_dataset, filtered_models)


def main():
    random.seed(13)
    np.random.seed(42)

    st.write('## Assignment of labels by labeling functions (LFs)')
    display_coverage()

    st.write('## Performance of models (bugginess)')
    filtered_models = display_bugginess_performance()

    st.write('## Debugging individual data points')
    display_individual_data_point_debugging(filtered_models)


main()

