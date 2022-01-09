from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

import numpy as np
import seaborn as sns
from dvc.exceptions import PathMissingError
from matplotlib import pyplot as plt
from pandas import MultiIndex
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(
    page_title="Bugginess",
    layout="wide",
)

from config import predefined_models, datasets_with_labels, datasets_without_labels, get_mnemonic_for_dataset
from data import read_labeled_dataset, get_dataset, compute_lf_coverages, compute_metric, get_fired_indexes, get_lfs

cm = sns.light_palette("green", as_cmap=True)


def datapoints_subset_ui(lfs, prefix, default_indices: Optional[Tuple[int, int]]=None):
    col1, col2, col3 = st.columns(3)
    col1.checkbox('Subset of data points', value=False, key=f'{prefix}.dataset_subset', help='Limit to those data points on which a specific LF was fired.')
    if st.session_state[f'{prefix}.dataset_subset']:
        lf = col2.selectbox('to which LF:', lfs, key=f'{prefix}.heuristic', index=default_indices[0] if default_indices else 0)
        value = col3.selectbox('assigned value:', [0, 1], key=f'{prefix}.value', index=default_indices[1] if default_indices else 0)
        return lf, value
    return None


def choose_dataset_ui(datasets, lfs, prefix, default_indices=None):
    chosen_dataset = st.selectbox('Select dataset to be analyzed:', datasets, key=f'{prefix}.msl', index=default_indices[0] if default_indices else 0, format_func=get_mnemonic_for_dataset)
    res = None
    if chosen_dataset is not None:
        res = datapoints_subset_ui(lfs, prefix, (default_indices[1], default_indices[2]))
    return chosen_dataset, res


def choose_datasets_ui(datasets, lfs, prefix):
    chosen_datasets = st.multiselect('Select dataset(s) to be analyzed:', options=datasets, default=datasets[:1], key=f'{prefix}.msl', format_func=get_mnemonic_for_dataset)
    res = None
    if len(chosen_datasets) > 0:
        res = datapoints_subset_ui(lfs, prefix)
    return chosen_datasets, res


def display_coverage(lfs):
    try:
        chosen_datasets, subset_criterion = choose_datasets_ui(datasets_with_labels + datasets_without_labels, lfs, 'lf_coverage')
        indices = None
        if subset_criterion is not None:
            indices = {dataset: get_fired_indexes(dataset, subset_criterion[0], subset_criterion[1]) for dataset in chosen_datasets}
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
                st_placeholder.info(f'There are no fired data points in any of the datasets. '
                                    f'Are you sure that LF `{subset_criterion[0]}` can assign value `{subset_criterion[1]}`?')
            st.download_button('Download', filtered_coverages_df.to_csv(), file_name='lf_values.csv')
    except PathMissingError as ex:
        st.warning("Selected dataset is not found. Was it uploaded to dvc remote?")
        st.exception(ex)


def display_model_filter_ui(filtered_models):
    col1, col2, col3, col4 = st.columns(4)

    col1.radio('Label source', ['keywords', 'all heuristics', 'gitcproc', 'gitcproc_orig', 'all'],
               key="label_source", index=4,
               help='keywords: only keyword heuristics\n\n'
                    'all heiuristics: keywords + file metrics + transformer trained on conventional commits\n\n'
                    'gitcproc:  9 "bugfix" keyword heuristics, "non-bugfix" if none matched\n\n'
                    'gitcproc_orig: 9 "bugfix" keyword heuristics, abstain if none matched')
    label_source = st.session_state.label_source
    if label_source != 'all':
        filtered_models = [model for model in filtered_models if predefined_models[model]['label_source'] == label_source]

    col2.radio('Issues', ['with issues', 'without issues', 'all'], key="issues", index=2)
    issues = st.session_state.issues
    if issues != 'all':
        filtered_models = [model for model in filtered_models if predefined_models[model]['issues'] == issues]

    col3.radio('Model', ['label model', 'transformer', 'all'], key="model", index=2)
    model1 = st.session_state.model
    if model1 != 'all':
        filtered_models = [model for model in filtered_models if predefined_models[model]['model'] == model1]

    if model1 in ['all', 'transformer']:
        col4.radio('Transformer trained on', ['only message', 'only change', 'message and change', 'all'], key="input_data", index=3)
        input_data = st.session_state['input_data']
        if input_data != 'all':
            filtered_models = [model for model in filtered_models if ('trained_on' not in predefined_models[model] or predefined_models[model]['trained_on'] == input_data)]
    return filtered_models


def display_model_perf_on_ind_data_points(models, dataset, indices, lf_name, lf_value):
    n_exp = len(models)
    for i, model in enumerate(models):
        raw_df = read_labeled_dataset(model, dataset, indices if indices is not None else None)
        if raw_df.empty:
            st.info('No datapoints found.')
            return
        if i == 0:
            if 'label' in raw_df.columns:
                prep_df: pd.DataFrame = raw_df[['sha', 'message', 'label']]
                prep_df.rename(columns={'label': 'true label'}, inplace=True)
            else:
                prep_df: pd.DataFrame = raw_df[['sha', 'message']]
        if 'ind_perf_metrics' not in st.session_state:
            st.session_state['ind_perf_metrics'] = 'accuracy'
        if st.session_state['ind_perf_metrics'] == 'accuracy':
            if 'label' in raw_df.columns:
                prep_df.loc[:, model] = 1 - (raw_df["prob_CommitLabel.BugFix"] - raw_df["label"]).abs()
            else:
                raise ValueError('Cannot calc accuracy for dataset without ground truth labels')
        elif st.session_state['ind_perf_metrics'] == 'probability':
            prep_df.loc[:, model] = raw_df["prob_CommitLabel.BugFix"]
        else:
            raise ValueError(f'Unknown value {st.session_state.ind_perf_metrics}')

    chosen_metric = st.session_state['ind_perf_metrics']
    if chosen_metric == 'accuracy':
        prep_df.loc[:, 'how_often_precise'] = prep_df[models].apply(lambda row: (row > 0.5).sum() / float(n_exp), axis=1)
        sort_columns=['how_often_precise', 'variance']
    elif chosen_metric == 'probability':
        precision = 1 - (prep_df[models].sub(prep_df["true label"], axis=0)).abs()
        prep_df.loc[:, 'how_often_precise'] = precision.apply(lambda row: (row > 0.5).sum() / float(n_exp), axis=1)
        sort_columns=['variance']
    else:
        raise ValueError(f'Unknown metric: {st.session_state["ind_perf_metrics"]}')
    prep_df.loc[:, 'variance'] = prep_df[models].apply(lambda row: np.var(row), axis=1)

    prep_df.sort_values(sort_columns, inplace=True, ascending=True)

    df_styler = prep_df.style.format({key: "{:.2%}" for key in list(models) + ['variance', 'how_often_precise']})
    df_styler = df_styler.background_gradient(cmap=cm)

    desc = format_subset_description(dataset, lf_name, lf_value, len(prep_df))
    st.write(desc)
    st.write(df_styler)
    st.radio('', ['accuracy', 'probability'], key='ind_perf_metrics', format_func=format_indiv_radio)
    st.download_button('Download', data=prep_df.to_csv(), file_name='preformance_data_points.csv')


def format_subset_description(dataset, lf_name, lf_value, n_datapoints):
    desc = f'`Displaying data points from dataset {dataset}'
    if lf_name is not None:
        desc += f' [{lf_name} == {lf_value}]'
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


def display_confusion_matrix(models, selected_datasets, indices, lf_name, lf_value):
    st.radio('Normalization', options=['None', 'true', 'pred', 'all'], key='normalize', format_func=format_normalization_labels)
    normalize = st.session_state['normalize']
    if normalize == 'None':
        normalize = None
    for dataset in selected_datasets:
        dataset_description = f'{get_mnemonic_for_dataset(dataset)}'
        if lf_name is not None:
            dataset_description += f'[{lf_name} == {lf_value}]'
        st.write(f"#### {dataset_description}")
        layout = get_layout_for_confusion(len(models))
        cols = [c for row in layout for c in st.columns(row)]
        for i, model in enumerate(models):
            confusion(dataset, model, indices, normalize, cols[i])


def display_model_metrics(models, selected_datasets, indices, lf_name, lf_value):
    if 'metric' not in st.session_state:
        st.session_state['metric'] = 'accuracy'
    if 'rel_imp' not in st.session_state:
        st.session_state['rel_imp'] = False
    if 'comp_mode' not in st.session_state:
        st.session_state['comp_mode'] = 'compare to baseline'
    matrix = []
    for model in models:
        row = []
        st.spinner(f'Computing metrics for model `{model}`')
        for dataset in selected_datasets:
            st.spinner(f'Computing metrics for dataset `{dataset}`')
            abstains_present = predefined_models[model]['model'] == 'label model'
            df = read_labeled_dataset(model, dataset, indices[dataset] if indices is not None else None)
            if not df.empty:
                res = compute_metric(df, st.session_state.metric, abstains_present)
            else:
                res = np.nan
            row.append(res)
        matrix.append(row)

    ind = MultiIndex.from_tuples([(
                                     model,
                                     predefined_models[model]['label_source'],
                                     predefined_models[model]['issues'],
                                     predefined_models[model]['model'],
                                     predefined_models[model]['trained_on'] if 'trained_on' in predefined_models[model] else 'N/A',
                                 ) for model in models], names=['id', 'label source', 'issues', 'model', 'trained on'])
    metrics_dataframe = pd.DataFrame(matrix, index=ind, columns=selected_datasets)
    metrics_dataframe = metrics_dataframe.sort_values(by=selected_datasets[0], inplace=False)
    if st.session_state.rel_imp:
        metrics_dataframe = convert_metrics_to_relative(metrics_dataframe)
    metrics_dataframe.reset_index(level=[1,2,3,4], inplace=True)
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
    st.checkbox('Show confiusion matrix(es)', value=False, key='conf_matrix_checkbox')
    if st.session_state['conf_matrix_checkbox']:
        display_confusion_matrix(models, selected_datasets, indices, lf_name, lf_value)


def display_datapoint_search_ui(dataset):
    st.text_input("Get full information about data point ...", key='rt', placeholder='Start typing a commit hash')
    if st.session_state.rt != '':
        datapoint = get_datapoint_by_id_beginning(dataset, st.session_state.rt)
        if datapoint:
            st.write(f'https://github.com/{datapoint["owner"]}/{datapoint["repo"]}/commit/{datapoint["_id"]}')
            st.write(datapoint)
        else:
            st.warning('Data point not found.')


def get_datapoint_by_id_beginning(dataset, id) -> Optional[Dict]:
    for datapoint in dataset:
        if datapoint['_id'].startswith(id):
            return datapoint
    return None


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
        10: [4, 4, 2],

    }
    return map[n]


def confusion(dataset, model, indices, normalize, handle):
    labeled_dataset = read_labeled_dataset(model, dataset, indices[dataset] if indices is not None else None)
    if labeled_dataset.empty:
        handle.write('No datapoints.')
        return
    predicted_continuous = labeled_dataset['prob_CommitLabel.BugFix']
    predicted = (predicted_continuous > 0.5).astype(int)
    ax= plt.subplot()
    disp = ConfusionMatrixDisplay.from_predictions(labeled_dataset['label'], predicted, normalize=normalize, ax=ax)
    title_details = "\n".join(predefined_models[model].values())
    ax.set_title(f'{model}\n\n{title_details}')
    handle.pyplot(disp.figure_)


def main():
    lfs = get_lfs(datasets_with_labels[-1])

    st.write('## Assignment of labels by labeling functions (LFs)')
    display_coverage(lfs)

    st.write('## Performance of models')

    filtered_models = display_model_filter_ui(predefined_models.keys())
    selected_datasets, subset_criterion = choose_datasets_ui(datasets_with_labels, lfs, 'model_perf')
    indices = None
    if subset_criterion is not None:
        indices = {dataset: get_fired_indexes(dataset, subset_criterion[0], subset_criterion[1]) for dataset in selected_datasets}
    if len(filtered_models) == 0:
        st.warning('No selected model satisfies chosen conditions.')
    elif len(selected_datasets) == 0:
        st.warning('No datasets selected.')
    else:
        lf_name, lf_value = subset_criterion if subset_criterion is not None else (None, None)
        display_model_metrics(filtered_models, selected_datasets, indices, lf_name, lf_value)

    st.write('## Debugging individual data points')
    default_indices = (
        datasets_with_labels.index(st.session_state[f'model_perf.msl'][0]) if (f'model_perf.msl' in st.session_state and st.session_state[f'model_perf.msl']) else 0,
        lfs.index(st.session_state[f'model_perf.heuristic']) if f'model_perf.heuristic' in st.session_state else 0,
        st.session_state[f'model_perf.value'] if f'model_perf.value' in st.session_state else 0,
    )
    selected_dataset, subset_criterion = choose_dataset_ui(datasets_with_labels, lfs, 'model_perf_ind', default_indices)
    indices = None
    lf_name = None
    lf_value = None
    if subset_criterion is not None:
        lf_name, lf_value = subset_criterion
        indices = get_fired_indexes(selected_dataset, lf_name, lf_value)
    if selected_dataset is not None and len(filtered_models) > 0:
        display_model_perf_on_ind_data_points(filtered_models, selected_dataset, indices, lf_name, lf_value)

        dataset = get_dataset(selected_dataset)
        display_datapoint_search_ui(dataset)


main()

