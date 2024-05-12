# general libraries
import tiktoken
import random
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os



# utils for running the chatbot
from efficiency.nlp import Chatbot


# local imports
from datasets.cpd_debates.cpd_debate_scraper import load_slices_dataset, create_slices_text_files, load_debates, token_counter, load_paragraphs
from .prompt_builder import SliceVariable, SingleSliceVariablePrompt, MultiSliceVariablePrompt, SpeakerVariable, MultiSpeakerVariableSingleSpeakerPrompt, MultiSpeakerVariablesMultiSpeakersPrompt, SingleSpeakerVariableSingleSpeakerPrompt, PerturbedSingleSpeakerVariableSingleSpeakerPrompt
from .observables import slice_variables, speaker_variables, speaker_variables_groups, slice_variables_groups, multi_speaker_variables_groups, slice_speaker_variables_correlated_by_design, predictor_variables, result_variables, speaker_result_variables_groups, speaker_predictor_variables_groups, contextual_variables_descriptions

# globals
model_name = 'gpt-3.5-turbo-0613'
model_output_folder = 'datasets/llm_measurements/'

cpd_dataset_folder = 'datasets/cpd_debates'
cpd_raw_debates_folder = f'{cpd_dataset_folder}/cpd_debates_raw'
cpd_slices_folder = f'{cpd_dataset_folder}/cpd_debates_slices'
cpd_slices_texts_folder = f'{cpd_dataset_folder}/cpd_debates_slices_text'
cpd_measurements_folder = f'{cpd_dataset_folder}/cpd_debates_measurements'


def get_slice_metadata(slices_d, slice_id):
    slice = slices_d[slice_id]

    metadata = {
        'slice_id': slice_id,
        'debate_id': slice['debate_id'],
        'num_tokens': slice['num_tokens'],
        'num_parts': slice['num_parts'],
        'speakers': slice['speakers'],
        'index': slice['index'],
        'slice_size': slice['slice_size'],
        'slice_overlap_ratio': slice['slice_overlap_ratio'],
        'slice_cutoff_ratio': slice['slice_cutoff_ratio'],
        'start_token_index': slice['start_token_index'],
        'end_token_index': slice['end_token_index'],
    }

    return metadata

def get_slice_contextual_variables(slices_d, slice_id):
    slice = slices_d[slice_id]

    vars = {
        'debate_id': slice['debate_id'],
        'num_tokens': slice['num_tokens'],
        'num_parts': slice['num_parts'],
        'slice_size': slice['slice_size'],
    }

    return vars

def get_speaker_metadata(slices_d, slice_id, speaker):
    slice = slices_d[slice_id]
    assert(speaker in slice['speakers'])

    speaker_metadata = {
        'speaker': speaker,
        'speaker_quantitative_contribution': slice['speakers_quantitative_contribution'][speaker],
        'speaker_quantitative_contribution_ratio': slice['speakers_quantitative_contribution_ratio'][speaker],
        'speaker_num_parts': slice['speakers_num_parts'][speaker],
        'speaker_num_parts_ratio': slice['speakers_num_parts_ratio'][speaker],
        **slice['defined_observables'][speaker]
    }

    return speaker_metadata

def get_speaker_contextual_variables(slices_d, slice_id, speaker):    
    slice = slices_d[slice_id]
    assert(speaker in slice['speakers'])

    do = slice['defined_observables'][speaker]

    speaker_contextual_variables = {
        'speaker_quantitative_contribution': slice['speakers_quantitative_contribution'][speaker],
        'speaker_quantitative_contribution_ratio': slice['speakers_quantitative_contribution_ratio'][speaker],
        'speaker_num_parts': slice['speakers_num_parts'][speaker],
        'speaker_num_parts_ratio': slice['speakers_num_parts_ratio'][speaker],
        
        'speaker_party': do['speaker_party'],
        'speaker_electoral_votes': do['speaker_electoral_votes'],
        'speaker_electoral_votes_ratio': do['speaker_electoral_votes_ratio'],
        'speaker_popular_votes': do['speaker_popular_votes'],
        'speaker_popular_votes_ratio': do['speaker_popular_votes_ratio'],
        'speaker_won_election': do['speaker_won_election'],
        'speaker_is_president_candidate': do['speaker_is_president_candidate'],
        'speaker_is_vice_president_candidate': do['speaker_is_vice_president_candidate'],
        'speaker_is_candidate': do['speaker_is_candidate'],
    }

    return speaker_contextual_variables

def get_speaker_slice_metadata(slices_d, slice_id, speaker):
    slice_metadata = get_slice_metadata(slices_d, slice_id)
    speaker_metadata = get_speaker_metadata(slices_d, slice_id, speaker)
    
    return {**slice_metadata, **speaker_metadata}

def get_contextual_slice_measurements(slices_d, slices):
    contextual_slice_measurements = []

    for slice in slices:
        slice_id = slice['id']

        for slice_variable, value in get_slice_contextual_variables(slices_d, slice_id).items():
            measurement = {
                'slice_id': slice_id,
                'name': slice_variable,
                'detailed_name': slice_variable,
                'value': value,
            }
            contextual_slice_measurements.append(measurement)
    
    return contextual_slice_measurements

def get_contextual_speaker_measurements(slices_d, slices):
    contextual_speaker_measurements = []

    for slice in slices:
        slice_id = slice['id']

        for speaker in slice['speakers']:
            for speaker_variable, value in get_speaker_contextual_variables(slices_d, slice_id, speaker).items():
                measurement = {
                    'slice_id': slice_id,
                    'speaker': speaker,
                    'name': speaker_variable,
                    'detailed_name': speaker_variable,
                    'value': value,
                }
                contextual_speaker_measurements.append(measurement)

    return contextual_speaker_measurements



# util function for loading the dataset
def load_dataset():
    debates = load_debates(cpd_raw_debates_folder, download_if_missing=True)

    slices = []

    for slice_size, slice_overlap_ratio, slice_cutoff_ratio in [
        # (250, 0.1, 0.05),
        # (500, 0.1, 0.05),
        # (1000, 0.1, 0.05),
        (2500, 0.1, 0.05),
        # (5000, 0.1, 0.05),
        # (12000, 0.1, 0.05),
    ]:
        # print(slice_size, int(slice_size*slice_cutoff_ratio))
        s = load_slices_dataset(debates=debates, slice_size=slice_size, slice_overlap_ratio=slice_overlap_ratio, slice_cutoff_ratio=slice_cutoff_ratio, slices_folder=cpd_slices_folder, model_name=model_name)
        create_slices_text_files(s, cpd_slices_texts_folder, slice_size=slice_size, slice_overlap_ratio=slice_overlap_ratio, slice_cutoff_ratio=slice_cutoff_ratio)
        slices += s

    paragraphs = load_paragraphs(slices)

    # assert that each debate, slice and paragraph has a unique id
    assert len(set([d['id'] for d in debates])) == len(debates)
    assert len(set([s['id'] for s in slices])) == len(slices)
    assert len(set([p['id'] for p in paragraphs])) == len(paragraphs)

    debates = {d['id']: d for d in debates}
    slices = {s['id']: s for s in slices}
    paragraphs = {p['id']: p for p in paragraphs}

    return debates, slices, paragraphs


def load_chatbot(model_name=model_name, cache_input_files=[], output_folder=model_output_folder, output_file_suffix=None):
    output_file = None
    if output_file_suffix is not None:
        output_file = f'{model_output_folder}/.cache_{model_name}_{output_file_suffix}.csv'
    bot = Chatbot(model_version=model_name, max_tokens=None, cache_files=cache_input_files, output_folder=output_folder, output_file=output_file)

    return bot

def slice_to_single_slice_variable_prompt(slice, observable):
    prompt = SingleSliceVariablePrompt(slice, observable)
    return prompt

def slices_to_single_slice_variable_prompt(slices, observable):
    prompts = [slice_to_single_slice_variable_prompt(slice, observable) for slice in slices]
    return prompts

def slice_and_speaker_to_single_speaker_observable_prompt(slice, speaker, observable):
    prompt = SingleSpeakerVariableSingleSpeakerPrompt(slice, speaker, observable)
    return prompt

def slices_and_speakers_to_single_speaker_variable_prompt(slices, observable, possible_speakers=None):
    prompts = []
    for slice in slices:
        for speaker in slice['speakers']:
            if possible_speakers is not None and speaker not in possible_speakers:
                continue
            prompt = slice_and_speaker_to_single_speaker_observable_prompt(slice, speaker, observable)
            prompts.append(prompt)
    return prompts

def slices_and_speaker_to_multi_speaker_variable_prompt(slices, observables, possible_speakers=None):
    prompts = []
    for slice in slices:
        for speaker in slice['speakers']:
            if possible_speakers is not None and speaker not in possible_speakers:
                continue
            prompt = MultiSpeakerVariableSingleSpeakerPrompt(slice, speaker, observables)
            prompts.append(prompt)
    return prompts


def slices_and_speakers_to_multi_speaker_variable_multi_speaker_prompt(slices, observables):
    prompts = []
    for slice in slices:
        speakers = slice['speakers']
        speakers = speakers.copy()
        assert(sorted(speakers) == speakers)
        random.seed(sum([sum([ord(c) for c in s]) for s in speakers]))
        random.shuffle(speakers)
        prompt = MultiSpeakerVariablesMultiSpeakersPrompt(slice, speakers, observables)
        prompts.append(prompt)
    return prompts



def slice_and_speaker_to_pertubation_single_speaker_observable_prompt(slice, speaker, variable, given_variable, given_value, pertubation, original_output_value):
    prompt = PerturbedSingleSpeakerVariableSingleSpeakerPrompt(slice, speaker, variable, given_variable, given_value, pertubation, original_output_value)
    return prompt



async def run_experiment(bot, prompts):
    # results = {}
    results = []

    print(f'Running {len(prompts)} prompts.')

    # batch_size = 1900 # 80
    batch_size = 2000
    batched_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    # for i, batch in tqdm(enumerate(batched_prompts)):
    for i, batch in enumerate(batched_prompts):
        print(f'\nBatch {i+1}/{len(batched_prompts)}')
        prompts = [prompt.get_prompt() for prompt in batch]
        responses = await bot._ask_n(prompts, num_parallel=60, temperature=0.0, verbose=0, pure_completion_mode=True)

        for response, prompt in zip(responses, batch):
            try:
                # prompt.parse_and_add_response(response, results)
                results += prompt.parse_response(response)
            except Exception as e:
                print(f'\t[ERROR] Failed to get response: {e}')
                print(f'\t[ERROR] Failed response: {response}')
                print('\n\n')
                # raise e

    # print bot cost
    print(f'\nBot cost:')
    bot.print_cost_and_rates()

    return results



# def save_slice_measurements(results, experiment_name):
#     if not os.path.exists(cpd_measurements_folder):
#         os.makedirs(cpd_measurements_folder)
#     if not os.path.exists(f'{cpd_measurements_folder}/slice_measurements'):
#         os.makedirs(f'{cpd_measurements_folder}/slice_measurements')
#     with open(f'{cpd_measurements_folder}/slice_measurements/{experiment_name}.pkl', 'wb') as f:
#         pickle.dump(results, f)

# def save_speaker_measurements(results, experiment_name):
#     if not os.path.exists(cpd_measurements_folder):
#         os.makedirs(cpd_measurements_folder)
#     if not os.path.exists(f'{cpd_measurements_folder}/speaker_measurements'):
#         os.makedirs(f'{cpd_measurements_folder}/speaker_measurements')
#     with open(f'{cpd_measurements_folder}/speaker_measurements/{experiment_name}.pkl', 'wb') as f:
#         pickle.dump(results, f)
