from typing import Dict, Optional

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

from config import dataset_mnemonic_to_id, predefined_models
from data import read_labeled_dataset, get_dataset, CHUNK_SIZE, compute_lf_coverages, get_label_matrix, \
    select_datapoints, compute_metric


cm = sns.light_palette("green", as_cmap=True)


def display_coverage(chosen_datasets):
    try:
        options = ['transformer', 'keyword', 'file metric']
        cols = st.columns(len(options))
        for col, option in zip(cols, options):
            col.checkbox(option, value=True, key=option)
        coverages_df = compute_lf_coverages([dataset_mnemonic_to_id[mn] for mn in chosen_datasets])
        lfs_to_include = [option for option in options if st.session_state[option]]
        filtered_coverages_df = coverages_df[coverages_df.index.isin(lfs_to_include, level=0)]
        df_styler = filtered_coverages_df.style.format({key: "{:.2%}" for key in filtered_coverages_df.columns})

        df_styler = df_styler.background_gradient(cmap=cm)
        st.write(df_styler)
        st.download_button('Download', filtered_coverages_df.to_csv(), file_name='lf_values.csv')
        return filtered_coverages_df
    except PathMissingError as ex:
        st.warning("Selected dataset is not found. Was it uploaded to dvc remote?")
        st.exception(ex)


def display_fired_datapoints(chosen_datasets):
    st.write('#### See fired datapoints')
    col1, col2, col3 = st.columns([3, 5, 2])
    dataset_mnemonic = col1.selectbox('Select dataset', chosen_datasets)
    label_matrix = get_label_matrix(dataset_mnemonic_to_id[dataset_mnemonic])
    heuristic = col2.selectbox('Select heuristic', label_matrix.columns)
    value = col3.selectbox('Select value', [0, 1])
    h_values = label_matrix[heuristic]
    h_values = h_values[h_values == value]
    dp_indexes = h_values.index.values.tolist()

    if f'batch.{dataset_mnemonic}' not in st.session_state:
        st.session_state[f'batch.{dataset_mnemonic}'] = 0

    def inc_batch():
        st.session_state[f'batch.{dataset_mnemonic}'] += 1

    def dec_batch():
        st.session_state[f'batch.{dataset_mnemonic}'] -= 1

    batch = st.session_state[f'batch.{dataset_mnemonic}']
    with st.spinner(f'Looking for data points on which LF `{heuristic}` was fired ...'):
        datapoints, dataset_truncated = select_datapoints(dataset_mnemonic, dp_indexes, batch)
    if dataset_truncated:
        st.warning(f'This dataset is large. Checking only data points {batch * CHUNK_SIZE} to {(batch+1) * CHUNK_SIZE}')
        if batch > 0:
            st.button(label=f'<< {(batch-1) * CHUNK_SIZE} - {(batch) * CHUNK_SIZE}', on_click=dec_batch)
        st.button(label=f'>> {(batch+1) * CHUNK_SIZE} - {(batch+2) * CHUNK_SIZE}', on_click=inc_batch)
    if len(datapoints) > 1:
        st.select_slider('Drag the slider to browse data points to which the selected labeling function assigns this value', key='sl', options=sorted(datapoints.keys()))

        index = st.session_state.sl
        st.json(datapoints[index])
    elif len(datapoints) == 1:
        st.info('Only one datapoint found:')
        st.json(datapoints[dp_indexes[0]])
    else:
        st.info('No datapoints found.')


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


def display_model_perf_on_ind_data_points(models, dataset):
    n_exp = len(models)
    for i, model in enumerate(models):
        df = read_labeled_dataset(model, dataset_mnemonic_to_id[dataset])
        if i == 0:
            res = df[['sha', 'message', 'label']]
        res.loc[:, f"{model}"] = df.apply(lambda row: 1 - np.abs(
            row["prob_CommitLabel.BugFix"] - row["label"]), axis = 1)

    res.set_index(['sha', 'message', 'label'], inplace=True)
    res.loc[:, 'variance'] = res.apply(lambda row: np.var(row), axis=1)
    res.loc[:, 'how_often_precise'] = res.apply(lambda row: (row > 0.5).sum() / float(n_exp), axis=1)
    res.sort_values(by=['how_often_precise', 'variance'], inplace=True, ascending=True)

    df_styler = res.style.format({key: "{:.2%}" for key in res.columns})

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


def display_model_metrics(models, selected_datasets):
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
            res = compute_metric(model, dataset_mnemonic_to_id[dataset], st.session_state.metric, abstains_present)
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

    metrics_styler = metrics_dataframe.style.format({key: "{:.2%}" for key in metrics_dataframe.columns})
    if st.session_state.rel_imp:
        metrics_styler = metrics_styler.background_gradient(cmap=sns.blend_palette("rg", as_cmap=True), vmin=-0.3, vmax=0.2, )
    else:
        metrics_styler = metrics_styler.background_gradient(cmap=cm)

    st.write(metrics_styler)
    st.download_button('Download', data=metrics_dataframe.to_csv(), file_name='metrics.csv')
    col1, col2 = st.columns(2)
    col1.radio('Metric', ['accuracy', 'f1', 'precision', 'recall', 'confusion matrix'], key='metric')
    col2.checkbox(label="Show relative improvement", value=False, key='rel_imp')
    if st.session_state.rel_imp:
        col2.radio('', ['compare to baseline', 'compare to previous model'], key='comp_mode')


def display_datapoint_search_ui(dataset):
    st.text_input("Search datapoint ...", key='rt', placeholder='Start typing a commit hash')
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
    st.write('## Assignment of labels by labeling functions')
    chosen_datasets = st.multiselect('Select datasets to be displayed:', dataset_mnemonic_to_id.keys(), key='msl')
    if len(chosen_datasets) > 0:
        display_coverage(chosen_datasets)
    display_fired_datapoints(dataset_mnemonic_to_id.keys())

    st.write('## Performance of models')

    # selected_models = st.multiselect('Select models: ', predefined_models.keys(), default=predefined_models.keys())
    filtered_models = display_model_filter_ui(predefined_models.keys())

    selected_datasets = st.multiselect('Select datasets:', dataset_mnemonic_to_id.keys())

    if len(filtered_models) == 0:
        st.warning('No selected model satisfies chosen conditions.')
    elif len(selected_datasets) == 0:
        st.warning('No datasets selected.')
    else:
        display_model_metrics(filtered_models, selected_datasets)

    st.write('#### Performance on individual data points')
    selected_dataset = st.selectbox('Select dataset:', dataset_mnemonic_to_id.keys())
    if selected_dataset is not None and len(filtered_models) > 0:
        display_model_perf_on_ind_data_points(filtered_models, selected_dataset)

        dataset = get_dataset(dataset_mnemonic_to_id[selected_dataset])
        display_datapoint_search_ui(dataset)


main()

