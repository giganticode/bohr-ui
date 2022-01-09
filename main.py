from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

import numpy as np
import seaborn as sns
from dvc.exceptions import PathMissingError
from pandas import MultiIndex

st.set_page_config(
    page_title="Bugginess",
    layout="wide",
)

from config import predefined_models, datasets_with_labels, datasets_without_labels, get_mnemonic_for_dataset
from data import read_labeled_dataset, get_dataset, compute_lf_coverages, compute_metric, get_fired_indexes, get_lfs

cm = sns.light_palette("green", as_cmap=True)


def datapoints_subset_ui(lfs, prefix, default_indices: Optional[Tuple[int, int]]=None):
    col1, col2, col3 = st.columns(3)
    col1.checkbox('Subset of data points', value=False, key=f'{prefix}.dataset_subset', help='!!!')
    if st.session_state[f'{prefix}.dataset_subset']:

        lf = col2.selectbox('Select LF', lfs, key=f'{prefix}.heuristic', index=default_indices[0] if default_indices else 0)
        value = col3.selectbox('Select value', [0, 1], key=f'{prefix}.value', index=default_indices[1] if default_indices else 0)
        return lf, value
    return None


def choose_dataset_ui(datasets, lfs, prefix, default_indices=None):
    chosen_dataset = st.selectbox('Select dataset to be displayed:', datasets, key=f'{prefix}.msl', index=default_indices[0] if default_indices else 0, format_func=get_mnemonic_for_dataset)
    indices = None
    if chosen_dataset is not None:
        res = datapoints_subset_ui(lfs, prefix, (default_indices[1], default_indices[2]))
        if res is not None:
            indices = get_fired_indexes(chosen_dataset, res[0], res[1])
    return chosen_dataset, indices


def choose_datasets_ui(datasets, lfs, prefix):
    chosen_datasets = st.multiselect('Select datasets to be displayed:', options=datasets, key=f'{prefix}.msl', format_func=get_mnemonic_for_dataset)
    indices = None
    if len(chosen_datasets) > 0:
        res = datapoints_subset_ui(lfs, prefix)
        if res is not None:
            indices = {dataset: get_fired_indexes(dataset, res[0], res[1]) for dataset in chosen_datasets}
    return chosen_datasets, indices


def display_coverage(lfs):
    try:
        chosen_datasets, indices = choose_datasets_ui(datasets_with_labels + datasets_without_labels, lfs, 'lf_coverage')
        if len(chosen_datasets) > 0:
            st_placeholder = st.empty()
            options = ['transformer', 'keyword', 'file metric']
            cols = st.columns(len(options))
            for col, option in zip(cols, options):
                col.checkbox(option, value=True, key=option)
            coverages_df = compute_lf_coverages([mn for mn in chosen_datasets], indices)
            lfs = coverages_df.columns
            lfs_to_include = [option for option in options if st.session_state[option]]
            filtered_coverages_df = coverages_df[coverages_df.index.isin(lfs_to_include, level=0)]

            filtered_coverages_df.reset_index(inplace=True)
            df_styler = filtered_coverages_df.style.format({key: "{:.2%}" for key in lfs})
            df_styler = df_styler.background_gradient(cmap=cm)
            if not filtered_coverages_df.empty:
                st_placeholder.write(df_styler)
                st.info('Labeling functions that have zero coverage throughout all the datasets are omited')
            else:
                st_placeholder.info('There are no fired data point in none of the datasets')
            st.download_button('Download', filtered_coverages_df.to_csv(), file_name='lf_values.csv')
            return filtered_coverages_df
    except PathMissingError as ex:
        st.warning("Selected dataset is not found. Was it uploaded to dvc remote?")
        st.exception(ex)


def display_model_filter_ui(filtered_models):
    col1, col2, col3, col4 = st.columns(4)

    col1.radio('Label source', ['keywords', 'all heuristics', 'gitcproc', 'gitcproc_orig', 'all'], key="label_source", index=3)
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


def display_model_perf_on_ind_data_points(models, dataset, indices):
    n_exp = len(models)
    index_columns = ['sha', 'message']
    for i, model in enumerate(models):
        df = read_labeled_dataset(model, dataset, indices if indices is not None else None)
        if df.empty:
            st.info('No datapoints found.')
            return
        if i == 0:
            if 'label' in df.columns:
                res: pd.DataFrame = df[['sha', 'message', 'label']]
                res.rename(columns={'label': 'true label'}, inplace=True)
                index_columns.append('true label')
            else:
                res: pd.DataFrame = df[['sha', 'message']]
        if 'label' in df.columns:
            res.loc[:, f"{model}"] = df.apply(lambda row: 1 - np.abs(
                row["prob_CommitLabel.BugFix"] - row["label"]), axis = 1)
        else:
            res.loc[:, f"{model}"] = df["prob_CommitLabel.BugFix"]

    res.set_index(index_columns, inplace=True)
    res.loc[:, 'variance'] = res.apply(lambda row: np.var(row), axis=1)
    res.loc[:, 'how_often_precise'] = res.apply(lambda row: (row > 0.5).sum() / float(n_exp), axis=1)
    res.sort_values(by=['how_often_precise', 'variance'], inplace=True, ascending=True)
    res.reset_index(inplace=True)

    df_styler = res.style.format({key: "{:.2%}" for key in list(models) + ['variance', 'how_often_precise']})

    df_styler = df_styler.background_gradient(cmap=cm)

    st.write(df_styler)
    st.download_button('Download', data=res.to_csv(), file_name='preformance_data_points.csv')


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


def display_model_metrics(models, selected_datasets, indices):
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
    col1, col2 = st.columns(2)
    col1.radio('Metric', ['accuracy', 'f1 (macro)', 'precision', 'recall', 'confusion matrix', 'certainty'], key='metric')
    col2.checkbox(label="Show relative improvement", value=False, key='rel_imp')
    if st.session_state.rel_imp:
        col2.radio('', ['compare to baseline', 'compare to previous model'], key='comp_mode')


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


def main():
    lfs = get_lfs(datasets_with_labels[-1])

    st.write('## Assignment of labels by labeling functions')
    display_coverage(lfs)

    st.write('## Performance of models')

    filtered_models = display_model_filter_ui(predefined_models.keys())
    selected_datasets, indices = choose_datasets_ui(datasets_with_labels, lfs, 'model_perf')
    if len(filtered_models) == 0:
        st.warning('No selected model satisfies chosen conditions.')
    elif len(selected_datasets) == 0:
        st.warning('No datasets selected.')
    else:
        display_model_metrics(filtered_models, selected_datasets, indices)

    st.write('## Debugging individual data points \n(Values that are shown are the probabilities assigned to true labels)')
    default_indices = (
        datasets_with_labels.index(st.session_state[f'model_perf.msl'][0]) if (f'model_perf.msl' in st.session_state and st.session_state[f'model_perf.msl']) else 0,
        lfs.index(st.session_state[f'model_perf.heuristic']) if f'model_perf.heuristic' in st.session_state else 0,
        st.session_state[f'model_perf.value'] if f'model_perf.value' in st.session_state else 0,
    )
    selected_dataset, indices = choose_dataset_ui(datasets_with_labels, lfs, 'model_perf_ind', default_indices)
    if selected_dataset is not None and len(filtered_models) > 0:
        display_model_perf_on_ind_data_points(filtered_models, selected_dataset, indices)

        dataset = get_dataset(selected_dataset)
        display_datapoint_search_ui(dataset)


main()

