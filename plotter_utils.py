import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import KFold
import itertools
import csv
from tqdm import tqdm
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, r_regression

# from main import load_dataset, speaker_observables_groups, slice_observables_groups, slice_speaker_observables_correlated_by_design, token_counter, predictor_observables, result_observables, load_slices_dataset, load_paragraphs, speaker_result_observables_groups, speaker_predictor_observables_groups, defined_observables_descriptions
from main import *

# globals
model_name = 'gpt-3.5-turbo-0613'
cpd_dataset_folder = 'datasets/cpd_debates'
cpd_raw_debates_folder = f'{cpd_dataset_folder}/cpd_debates_raw'
cpd_slices_folder = f'{cpd_dataset_folder}/cpd_debates_slices'
cpd_slices_texts_folder = f'{cpd_dataset_folder}/cpd_debates_slices_text'
cpd_measurements_folder = f'{cpd_dataset_folder}/cpd_debates_measurements'
plots_folder = 'plots'

report_latex_folder = 'report/sec/code_info/'


# estimated total cost
def print_estimated_cost():
    with open(f'.cache_{model_name}.csv', 'r') as f:
        reader = csv.reader(f)
        # remove header
        next(reader) # pred, query
        assert('gpt-3.5-turbo' in model_name) # otherwise prices wrong...
        input_token_cost = 0.0015
        output_token_cost = 0.002
        total_cost = 0
        total_queries = 0
        total_in_tokens = 0
        total_out_tokens = 0
        total_in_words = 0
        total_out_words = 0
        total_characters = 90000
        words_per_minute = 175
        for row in tqdm(list(reader)):
            total_queries += 1
            total_in_tokens += token_counter(row[1], model_name)
            total_out_tokens += token_counter(row[0], model_name)
            total_in_words += len(row[1].split())
            total_out_words += len(row[0].split())
        total_tokens = total_in_tokens + total_out_tokens
        total_in_cost = total_in_tokens * input_token_cost / 1000
        total_out_cost = total_out_tokens * output_token_cost / 1000
        total_cost = total_in_cost + total_out_cost
        total_words = total_in_words + total_out_words

    print('OpenAI API usage statistics:')
    print(f'\tTotal queries: {total_queries:,}')
    print(f'\tTotal tokens: {total_tokens:,}')
    print(f'\t\tInput tokens: {total_in_tokens:,}')
    print(f'\t\tOutput tokens: {total_out_tokens:,}')
    print(f'\t\tCompared to whole English Wikipedia: {total_tokens / 6000000000 * 100:.3f}%')
    print(f'\tTotal Word Count: {total_words:,}')
    print(f'\t\tInput words: {total_in_words:,}')
    print(f'\t\tOutput words: {total_out_words:,}')
    print(f'\tEstimated speaking time (175 words per minute, which is fast): {total_words / words_per_minute / 60:.2f} hours')
    print(f'\tTotal cost: {total_cost:,.2f} USD')
    print(f'\t\tInput cost: {total_in_cost:,.2f} USD')
    print(f'\t\tOutput cost: {total_out_cost:,.2f} USD')
    print(f'\tEstimated human cost (assuming 20 $/h): {total_words / words_per_minute / 60 * 20:.2f} USD')

def print_dataset_statistics():
    # dataset used in main
    debates_d, slices_d, paragraphs_d = load_dataset()

    _s = load_slices_dataset(debates=debates_d.values(), slice_size=2500, slice_overlap_ratio=0.1, slice_cutoff_ratio=0.05, slices_folder=cpd_slices_folder, model_name=model_name)
    _paragraphs = load_paragraphs(_s)

    # dataset stats:
    print('Dataset statistics:')
    print(f'\tDebates: {len(debates_d)}')
    print(f'\tSlices: {len(_s)}')
    print(f'\tParagraphs: {len(_paragraphs)}')
    print(f'\tTokens: {sum([s["num_tokens"] for s in _s]):,}')
    print(f'\tWords: {sum([len(s["text"].split()) for s in _s]):,}')
    print(f'\tSentences: {sum([len(s["text"].split(".")) for s in _s]):,}')
    print(f'\tEstimated speaking time (175 words per minute, which is fast): {sum([len(s["text"].split()) for s in _s]) / 175 / 60:.2f} hours')



# utils for loading data
def load_slice_measurements():
    measurements = {}
    for filename in os.listdir(f'{cpd_measurements_folder}/slice_measurements'):
        if filename.endswith('.pkl'):
            with open(f'{cpd_measurements_folder}/slice_measurements/{filename}', 'rb') as f:
                measurements[filename.rsplit('.', 1)[0]] = pickle.load(f)
    return measurements

def load_speaker_measurements():
    measurements = {}
    for filename in os.listdir(f'{cpd_measurements_folder}/speaker_measurements'):
        if filename.endswith('.pkl'):
            with open(f'{cpd_measurements_folder}/speaker_measurements/{filename}', 'rb') as f:
                measurements[filename.rsplit('.', 1)[0]] = pickle.load(f)
    return measurements

def load_slice_and_measured_observables(slices_d):
    slice_measurements = load_slice_measurements()
    assert(len(slice_measurements) == 1)
    slice_measurements = slice_measurements[list(slice_measurements.keys())[0]]
    slice_observables = {}

    for slice_name, slice_measurement in slice_measurements.items():
        slice_defined_observables = slices_d[slice_name]['defined_observables']
        slice_measured_observables = slice_measurement
        for speaker in slice_defined_observables:
            speaker_defined_observables = slice_defined_observables[speaker]

            # check that no key overlap
            assert(set(speaker_defined_observables.keys()).isdisjoint(set(slice_measured_observables.keys())))
            
            # merge dictionaries
            slice_observables[slice_name] = slice_observables.get(slice_name, {})
            slice_observables[slice_name][speaker] = {**speaker_defined_observables, **slice_measured_observables}

    return slice_observables

def transform_str_categories_to_flags(df, column_name):
    df = df.copy()
    categories = df[column_name].unique()
    for category in categories:
        df[f'{column_name}_is_{category}'] = df[column_name].apply(lambda x: 1 if x == category else 0)
    df.drop(columns=[column_name], inplace=True)
    return df

def create_speaker_df(slice_speaker_base_observables, slice_speaker_observables):
    entries = []
    for slice_name, slice_speaker_observable in slice_speaker_observables.items():
        for speaker, speaker_observable in slice_speaker_observable.items():
            assert(set(slice_speaker_base_observables[slice_name][speaker].keys()).isdisjoint(set(speaker_observable.keys())))
            entry = {**slice_speaker_base_observables[slice_name][speaker], **speaker_observable}
            entries.append(entry)
    return pd.DataFrame(entries)

def create_grouped_observables_df(df):
    df = df.copy()
    for name, group in speaker_observables_groups.items():
        group_detailed_names = [g.detailed_name for g in group if g.detailed_name in df.columns]
        if len(group_detailed_names) == 0:
            continue
        df[name] = df[group_detailed_names].mean(axis=1)
        df.drop(columns=group_detailed_names, inplace=True)
    
    for name, group in slice_observables_groups.items():
        group_detailed_names = [g.detailed_name for g in group if g.detailed_name in df.columns]
        if len(group_detailed_names) == 0:
            continue
        if isinstance(df[group_detailed_names[0]][0], str):
            if len(group_detailed_names) == 1:
                df[name] = df[group_detailed_names[0]]
            else:
                assert(False)
            df[name] = df[group_detailed_names].apply(lambda x: ' '.join(x), axis=1)
        else:
            df[name] = df[group_detailed_names].mean(axis=1)
        df.drop(columns=group_detailed_names, inplace=True)
    
    return df

def prepare_raw_speaker_observables(slice_speaker_base_observables, slice_speaker_observables):
    df = create_speaker_df(slice_speaker_base_observables, slice_speaker_observables)
    
    # add num_entries per speaker as a column
    df['speaker_num_entries_in_dataset'] = df['speaker'].map(df['speaker'].value_counts())

    observable_means = np.array([df[observable].mean() for observable in df.columns if not isinstance(df[observable].iloc[0], str)])
    observable_stds = np.array([df[observable].std() for observable in df.columns if not isinstance(df[observable].iloc[0], str)])
    # replace 0 in observable_stds with 1
    observable_stds[observable_stds == 0] = 1

    speakers_means = {speaker: [df[observable][df['speaker'] == speaker].mean() for observable in df.columns if not isinstance(df[observable].iloc[0], str)] for speaker in df['speaker'].unique()}
    # add outlier score that is square root of normalized observables for each speaker
    df['outlier_score'] = df['speaker'].map(lambda x: ((speakers_means[x]-observable_means)/observable_stds).dot((speakers_means[x]-observable_means)/observable_stds))
    # normalize outlier score
    outlier_score_mean = df['outlier_score'].mean()
    outlier_score_std = df['outlier_score'].std()
    df['outlier_score'] = df['outlier_score'].map(lambda x: (x-outlier_score_mean)/outlier_score_std)

    return df

def prepare_speaker_observables(slice_speaker_base_observables, slice_speaker_observables):
    df = prepare_raw_speaker_observables(slice_speaker_base_observables, slice_speaker_observables)

    n_raw_observables = len(df.columns)

    df = create_grouped_observables_df(df)

    n_grouped_observables = len(df.columns)

    df = transform_str_categories_to_flags(df, 'speaker_party')
    df = transform_str_categories_to_flags(df, 'debate_elected_party')

    return df, n_raw_observables, n_grouped_observables



def save_observables_to_latex_list(observable_group, groups_name, caption, label):
    s = []
    # for name, val in observable_group.items():
    #     s += [f'\\\\\n\t{name} & {val[0].datatype} \\\\\n\t\hline']
    #     for observable in val:
    #         s += [f'\t\t{observable.detailed_name.split("(")[-1].split(")")[0]} & {observable.description} \\\\']
    #     s += ['\hline']
    for name, val in observable_group.items():
        s += [f'\t\\SetCell{{bg=blue9}} \\textbf{{{name}}} & \\SetCell{{bg=blue9}} {val[0].datatype} \\\\']
        for observable in val:
            s += [f'\t\t{observable.detailed_name.split("(")[-1].split(")")[0]} & {observable.description} \\\\']
        s += ['\hline']

    s = ['\hline']+s[:-1]
    s = '\n'.join(s)
    s = s.replace('_', '\\_')
    # for name, val in slice_observables_groups.items():
    #     s += [f'\t\item {name}']
    #     for observable in val:
    #         s += [f'\t\t\subitem {observable.detailed_name} <{observable.datatype}>: {observable.description}']
    #         # s += [f'\t\t\subitem detailed_name:\t{observable.detailed_name}']
    #         # s += [f'\t\t\subitem type:\t{observable.detailed_name}']
    #         # s += [f'\t\t\subitem description:\t{observable.description}']

    # s = '\n'.join(s)
    # s = s.replace('_', '\\_')

    st = """\\begin{{longtblr}}[
    caption = {{{caption}}},
    label = {{{label}}},
]{{
    colspec = {{Q[2.2cm] Q[1]}},
    rowhead = 1,
    hlines,
    vlines,
    row{{1}} = {{olive9}},
}}
Group, Name & Description \\\\"""
    s = st.format(caption=caption, label=label) + '\n' + s + '\n' + '\end{longtblr}'

    with open(f'{report_latex_folder}/{groups_name}.tex', 'w') as f:
        f.write(s)

def save_given_observables(slices_d, caption, label, subset=None, file_name='defined_observables_description'):
    global defined_observables_descriptions

    defined_observables = list(list(slices_d.values())[0]['defined_observables'].values())[0]
    for do in defined_observables:
        # check if new defined observables that are not in description
        assert(do in defined_observables_descriptions)
    s = []
    for name, des in defined_observables_descriptions.items():
        if subset is not None and name not in subset:
            continue
        s += [f'\t{name} & {des} \\\\']
    s = '\n'.join(s)
    s = s.replace('_', '\\_ ')

    st = """\\begin{{longtblr}}[
    caption = {{{caption}}},
    label = {{{label}}},
]{{
    colspec = {{Q[2.2cm] Q[1]}},
    rowhead = 1,
    hlines,
    vlines,
    row{{even}} = {{gray9}},
    row{{1}} = {{olive9}},
}}
Name & Description \\\\"""
    s = st.format(caption=caption, label=label) + '\n' + s + '\n' + '\end{longtblr}'

    with open(f'{report_latex_folder}/{file_name}.tex', 'w') as f:
        f.write(s)

def kfold_cross_validation(df, accumulation_func, combine_func, n_splits=2, iterations=50):
    _vals = []
    for i in range(iterations):
        kf = KFold(n_splits, shuffle=True, random_state=i)
        
        for train_index, _ in kf.split(df):
            train_data = df.iloc[train_index]
            _vals.append(accumulation_func(train_data))
    
    columns = _vals[0].columns
    rows = _vals[0].index

    vals = np.array(_vals)
    acc_vals = np.zeros_like(vals[0])
    for i, j in itertools.product(range(vals.shape[1]), range(vals.shape[2])):
        acc_vals[i, j] = combine_func(vals[:, i, j])

    acc_vals = pd.DataFrame(acc_vals, index=rows, columns=columns)

    return acc_vals