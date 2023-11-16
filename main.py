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
from prompt_builder import SliceObservable, SingleSliceObservablePrompt, MultiSliceObservablePrompt, SpeakerObservable, MultiSpeakerObservablePrompt, MultiSpeakerObservableMultiSpeakersPrompt
from observables import slice_observables, speaker_observables, speaker_observables_groups, slice_observables_groups, multi_speaker_observables_groups, slice_speaker_observables_correlated_by_design, predictor_observables, result_observables, speaker_result_observables_groups, speaker_predictor_observables_groups, defined_observables_descriptions



# globals
model_name = 'gpt-3.5-turbo-0613'
cpd_dataset_folder = 'datasets/cpd_debates'
cpd_raw_debates_folder = f'{cpd_dataset_folder}/cpd_debates_raw'
cpd_slices_folder = f'{cpd_dataset_folder}/cpd_debates_slices'
cpd_slices_texts_folder = f'{cpd_dataset_folder}/cpd_debates_slices_text'
cpd_measurements_folder = f'{cpd_dataset_folder}/cpd_debates_measurements'


# util function for loading the dataset
def load_dataset():
    debates = load_debates(cpd_raw_debates_folder, download_if_missing=True)

    slices = []

    for slice_size, slice_overlap_ratio, slice_cutoff_ratio in [
        (250, 0.1, 0.05),
        (500, 0.1, 0.05),
        (1000, 0.1, 0.05),
        (2500, 0.1, 0.05),
        (5000, 0.1, 0.05),
        (12000, 0.1, 0.05),
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



def load_chatbot():
    model_name = 'gpt-3.5-turbo-0613'
    bot = Chatbot(model_version=model_name, max_tokens=None)

    return bot

def slice_to_single_slice_observable_prompt(slice, observable):
    prompt = SingleSliceObservablePrompt(slice, observable)
    return prompt

def slices_to_single_slice_observable_prompt(slices, observable):
    prompts = [slice_to_single_slice_observable_prompt(slice, observable) for slice in slices]
    return prompts

def slice_and_speaker_to_single_speaker_observable_prompt(slice, speaker, observable):
    prompt = MultiSpeakerObservablePrompt(slice, speaker, [observable])
    return prompt

def slices_and_speakers_to_single_speaker_observable_prompt(slices, observable, possible_speakers=None):
    prompts = []
    for slice in slices:
        for speaker in slice['speakers']:
            if possible_speakers is not None and speaker not in possible_speakers:
                continue
            prompt = slice_and_speaker_to_single_speaker_observable_prompt(slice, speaker, observable)
            prompts.append(prompt)
    return prompts

def slices_and_speaker_to_multi_speaker_observable_prompt(slices, observables, possible_speakers=None):
    prompts = []
    for slice in slices:
        for speaker in slice['speakers']:
            if possible_speakers is not None and speaker not in possible_speakers:
                continue
            prompt = MultiSpeakerObservablePrompt(slice, speaker, observables)
            prompts.append(prompt)
    return prompts


def slices_and_speakers_to_multi_speaker_observable_multi_speaker_prompt(slices, observables):
    prompts = []
    for slice in slices:
        speakers = slice['speakers']
        speakers = speakers.copy()
        assert(sorted(speakers) == speakers)
        random.seed(sum([sum([ord(c) for c in s]) for s in speakers]))
        random.shuffle(speakers)
        prompt = MultiSpeakerObservableMultiSpeakersPrompt(slice, speakers, observables)
        prompts.append(prompt)
    return prompts



def run_experiment(bot, prompts):
    results = {}

    print(f'Running {len(prompts)} prompts.')

    # batch_size = 1900 # 80
    batch_size = 2000
    batched_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    # for i, batch in tqdm(enumerate(batched_prompts)):
    for i, batch in enumerate(batched_prompts):
        print(f'\nBatch {i+1}/{len(batched_prompts)}')
        prompts = [prompt.get_prompt() for prompt in batch]
        responses = bot.ask_n(prompts, num_parallel=60, temperature=0.0, verbose=0, pure_completion_mode=True)

        for response, prompt in zip(responses, batch):
            try:
                prompt.parse_and_add_response(response, results)
            except Exception as e:
                print(f'\t[ERROR] Failed to get response: {e}')
                print(f'\t[ERROR] Failed response: {response}')
                print('\n\n')
                # raise e

    # print bot cost
    print(f'\nBot cost:')
    bot.print_cost_and_rates()

    return results



def save_slice_measurements(results, experiment_name):
    if not os.path.exists(cpd_measurements_folder):
        os.makedirs(cpd_measurements_folder)
    if not os.path.exists(f'{cpd_measurements_folder}/slice_measurements'):
        os.makedirs(f'{cpd_measurements_folder}/slice_measurements')
    with open(f'{cpd_measurements_folder}/slice_measurements/{experiment_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

def save_speaker_measurements(results, experiment_name):
    if not os.path.exists(cpd_measurements_folder):
        os.makedirs(cpd_measurements_folder)
    if not os.path.exists(f'{cpd_measurements_folder}/speaker_measurements'):
        os.makedirs(f'{cpd_measurements_folder}/speaker_measurements')
    with open(f'{cpd_measurements_folder}/speaker_measurements/{experiment_name}.pkl', 'wb') as f:
        pickle.dump(results, f)



def main():
    # possible_speakers = all presidential candidates
    possible_speakers = None # if None then all, otherwise a white list
    # num_paragraphs = 10
    num_slices = 150
    # num_slices = 80
    num_special_slices = 50 # how many slices to use for special experiments


    bot = load_chatbot()


    debates_d, slices_d, paragraphs_d = load_dataset()
    print(f'Loaded:\n\tDebates: {len(debates_d)}\n\tSlices: {len(slices_d)}\n\tParagraphs: {len(paragraphs_d)}')

    # remove all that are too long
    debates = list(debates_d.values())
    # slices = [slice for slice in slices_d.values() if slice['slice_size'] == 250 or slice['slice_size'] == 500]
    slices = [slice for slice in slices_d.values() if slice['slice_size'] == 2500]
    paragraphs = load_paragraphs(slices)

    print(f'Only keeping slice_size=2500:\n\tDebates: {len(debates)}\n\tSlices: {len(slices)}\n\tParagraphs: {len(paragraphs)}')

    # only taking subset of paragraphs at random
    random.seed(42)
    random.shuffle(slices)
    slices = slices[:num_slices]
    
    print(f'Looking at subset:\n\tDebates: {len(debates)}\n\tSlices: {len(slices)}\n\tParagraphs: {len(paragraphs)}')
    

    # count speaker occurrences
    from collections import Counter
    speakers = Counter()
    for slice in slices:
        speakers.update(slice['speakers'])
    print('\n')
    print(f'Found {len(speakers)} distinct speakers.')
    print(f'Found {sum(speakers.values())} total speaker occurrences.')
    print(speakers)



    collected_measurements = {}

    print("\n\n\nSINGLE [SLICE OBSERVABLE] EXPERIMENT")
    prompts = []
    for slice_observable in slice_observables.values():
        prompts += slices_to_single_slice_observable_prompt(slices, slice_observable)

    SSLO_measurements = run_experiment(bot, prompts)
    print(list(SSLO_measurements.values())[0].keys())
    save_slice_measurements(SSLO_measurements, 'SSLO_measurements')
    collected_measurements['SSLO'] = SSLO_measurements


    print("\n\n\nWHOLE SLICE, SINGLE SPEAKER, SINGLE [SPEAKER OBSERVABLE] EXPERIMENT")
    prompts = []
    for speaker_observable in speaker_observables.values():
        prompts += slices_and_speakers_to_single_speaker_observable_prompt(slices, speaker_observable, possible_speakers=possible_speakers)

    WS_SS_SSO_measurements = run_experiment(bot, prompts)
    print(list(list(WS_SS_SSO_measurements.values())[0].values())[0].keys())
    save_speaker_measurements(WS_SS_SSO_measurements, 'WS_SS_SSO_measurements')
    collected_measurements['WS_SS_SSO'] = WS_SS_SSO_measurements

    slices = slices[:num_special_slices]

    print("\n\n\nWHOLE SLICE, MULTI SPEAKER, SINGLE [SPEAKER OBSERVABLE] EXPERIMENT")
    prompts = []
    for speaker_observable in speaker_observables.values():
        prompts += slices_and_speakers_to_multi_speaker_observable_multi_speaker_prompt(slices, [speaker_observable])
    
    WS_MS_SSO_measurements = run_experiment(bot, prompts)
    print(list(list(WS_MS_SSO_measurements.values())[0].values())[0].keys())
    save_speaker_measurements(WS_MS_SSO_measurements, 'WS_MS_SSO_measurements')
    collected_measurements['WS_MS_SSO'] = WS_MS_SSO_measurements


    print("\n\n\nWHOLE SLICE, SINGLE SPEAKER, MULTI [SPEAKER OBSERVABLE] EXPERIMENT")
    WS_SS_MSO_measurements = {}
    for key, multi_speaker_observables_group in multi_speaker_observables_groups.items():
        prompts = slices_and_speaker_to_multi_speaker_observable_prompt(slices, multi_speaker_observables_group)
    
        WS_SS_MSO_measurements[key] = run_experiment(bot, prompts)
        print(f'{key}:\t{list(list(WS_SS_MSO_measurements.values())[0].values())[0].keys()}')
        save_speaker_measurements(WS_SS_MSO_measurements[key], f'WS_SS_MSO_{key}_measurements')
    collected_measurements['WS_SS_MSO'] = WS_SS_MSO_measurements

if __name__ == '__main__':
    main()
