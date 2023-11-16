# 1. get all links from: https://www.debates.org/voter-education/debate-transcripts/ under id='content-sm'
# 2. open each link and download the text under id='content-sm'
# 3. save each text as a separate file in a folder called 'cpd_debates_raw'

import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from time import sleep
import regex as re
import pandas as pd
import tiktoken
import pickle
import shutil



def encoding_getter(encoding_type: str):
    """
    Returns the appropriate encoding based on the given encoding type (either an encoding string or a model name).
    """
    if "k_base" in encoding_type:
        return tiktoken.get_encoding(encoding_type)
    else:
        return tiktoken.encoding_for_model(encoding_type)

def tokenizer(string: str, encoding_type: str) -> list:
    """
    Returns the tokens in a text string using the specified encoding.
    """
    encoding = encoding_getter(encoding_type)
    tokens = encoding.encode(string)
    return tokens

def detokenizer(string: str, encoding_type: str) -> list:
    """
    Returns the tokens in a text string using the specified encoding.
    """
    encoding = encoding_getter(encoding_type)
    tokens = encoding.decode(string)
    return tokens

def token_counter(string: str, encoding_type: str) -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.
    """
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens



def get_links():
    url = 'https://www.debates.org/voter-education/debate-transcripts/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find(id='content-sm').find_all('a')
    links = [link['href'] for link in links]

    # if doesn't start with https, add https://www.debates.org
    links = [link if link.startswith('http') else 'https://www.debates.org' + link for link in links]

    # remove if contains translation but print INFO
    for link in links:
        if 'translation' in link:
            print(f'INFO: Removing {link} from links.')
            links.remove(link)

    return links

def get_text(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find(id='content-sm').get_text()
    return text

def save_text(text, filename):
    with open(filename, 'w') as f:
        f.write(text)

def download_debates(folder='cpd_debates_raw'):
    links = get_links()
    print(f'Found {len(links)} links.')

    for link in tqdm(links):
        text = get_text(link)
        if link[-1] == '/':
            link = link[:-1]
        filename = link.split('/')[-1]
        save_text(text, os.path.join(folder, filename))
        sleep(1)



def parse_metadata(raw_meta: str):
    """
    turns metadata into dict
    """
    meta = {}

    # parse date from meta in form of "October 3, 2000"
    date_match = re.search(r'(?P<month>[A-Z][a-z]+) (?P<day>[0-9]+), (?P<year>[0-9]+)', raw_meta)
    meta['date'] = date_match.groupdict()

    from .past_presidents import year_to_presidents

    meta['election_data'] = year_to_presidents[meta['date']['year']]

    # dict for mapping short name to full name and president to full name
    year = meta['date']['year']
    candidate_year_mapping = {k.split(' ')[-1]: k for k in year_to_presidents[year]['president_candidates'].values()}
    candidate_year_mapping.update({k.split(' ')[-1]: k for k in year_to_presidents[year]['vice_president_candidates'].values()})

    candidate_party_mapping = {v: k for k, v in year_to_presidents[year]['president_candidates'].items()}
    candidate_party_mapping.update({v: k for k, v in year_to_presidents[year]['vice_president_candidates'].items()})

    # add president to name mapping using year - 4
    pyear = str(int(year) - 4)
    if pyear=='1956':
        candidate_year_mapping.update({
            'PRESIDENT': 'DWIGHT EISENHOWER',
            'VICE PRESIDENT': 'RICHARD NIXON',
        })
    else:
        candidate_year_mapping.update({
            'PRESIDENT': year_to_presidents[pyear]['elected_president'],
            'VICE PRESIDENT': year_to_presidents[pyear]['elected_vice_president'],
        })
    
    # possible_speakers = list(set([e['president_candidates'].values() for e in year_to_presidents.values()] + [e['vice_president_candidates'].values() for e in year_to_presidents.values()]))
    possible_speakers = [list(e['president_candidates'].values()) + list(e['vice_president_candidates'].values()) for e in year_to_presidents.values()]
    possible_speakers = [item for sublist in possible_speakers for item in sublist]
    possible_speakers = list(set(possible_speakers))

    meta['possible_speakers'] = possible_speakers
    meta['candidate_year_mapping'] = candidate_year_mapping
    meta['candidate_party_mapping'] = candidate_party_mapping

    return meta

def parse_debate(_raw_debate: str, debate_name, debate_index):
    """
    turns debate into metadata and csv
    """

    # remove everything after "END", especially since their is a dublicate in the 2008 for some reason...
    _raw_debate = _raw_debate.split('\nEND')[0].strip()
    
    debate = _raw_debate
    # fix wrong speaker names
    mistakes = {
        # misspellings
        'OBAM': 'OBAMA',
        'ROMNEHY': 'ROMNEY',
        'ADMIRAL STOCKDALE': 'STOCKDALE',
        'HAL BRUNO': 'BRUNO',
        'VICE PRESIDENT QUAYLE': 'QUAYLE',
        'SENATOR GORE': 'GORE',
        'SM1TH': 'SMITH',
        'PRESDIENT BUSH': 'BUSH',
        # should be uniform accross all debates
        'THE PRESIDENT': 'PRESIDENT',
        'QUESTION': 'AUDIENCE QUESTION',
        # these cause problems with parsing
        'MR.': '',
        'MS.': '',
        'MRS.': '',
        'Mr.': '',
        'Ms.': '',
        'Mrs.': '',
        # these cause problems with parsing
        '(CROSSTALK) ': '(CROSSTALK)\n',
        '(LAUGHTER) ': '(LAUGHTER)\n',
        '[*]': '\n',
        '[*] ': '\n',
        '\xa0': ' ',
        ' (continuing):': ': (continuing)',

        # just small fixes
        '  ': ' ',
    }
    for mistake in mistakes:
        debate = debate.replace(mistake, mistakes[mistake])
        # check that not OBAMA -> OBAMAA should be OBAM -> OBAMA and OBAMA -> OBAMA
        if mistake in mistakes[mistake]:
            double_replace = mistakes[mistake].replace(mistake, mistakes[mistake])
            # print(f'|{double_replace}|\t|{mistakes[mistake]}|')
            debate = debate.replace(double_replace, mistakes[mistake])
    
    # remove unnecessary lines
    lines = debate.split('\n')
    lines = [line for line in lines if 
        'Transcribed by: ' not in line and
        'Transcription by: ' not in line]
    debate = '\n'.join(lines)
    
    # split into metadata and debate at first line containing `: `
    split = debate.split('\n')
    for i, line in enumerate(split):
        if ': ' in line:
            break
    result = parse_metadata('\n'.join(split[:i]))
    result['name'] = debate_name
    result['index'] = debate_index
    result['id'] = f'di{debate_index}'

    debate = '\n'.join(split[i:]).strip()

    # improve the uniqueness of names for a given year by avoiding clinton mixups
    candidate_year_mapping = result.pop('candidate_year_mapping')
    
    for name in candidate_year_mapping:
        debate = debate.replace(name, candidate_year_mapping[name])
    
    # fix double replacement, for example `BARACK OBAMA` -> `BARACK BARACK OBAMA`
    for name in candidate_year_mapping:
        debate = debate.replace(name + ' ' + candidate_year_mapping[name], candidate_year_mapping[name])

    # get all speakers of form `some name that migth include a comma or other punctuation: `
    speakers = re.findall(r'(?P<speaker>[A-Zcrs \.\,\`\’]+): ', debate)
    speakers = [s.strip() for s in speakers]

    # fix missing \n before speaker
    debate.replace('  ', ' ')
    debate.replace('  ', ' ')
    debate.replace('  ', ' ')
    for speaker in speakers:
        debate = debate.replace(f'. {speaker}: ', f'.\n{speaker}: ')
        debate = debate.replace(f'\n {speaker}: ', f'\n{speaker}: ')

    # split into actual debate
    parts = []
    for line in debate.split('\n'):
        found = False
        for speaker in speakers:
            if line.startswith(f'{speaker}: '):
                parts.append({
                    'index': len(parts),
                    'debate_id': result['id'],
                    'speaker': speaker,
                    'text': line[len(speaker)+2:],
                    'id': f'{result["id"]}-{len(parts)}',
                })
                found = True
                break
        if not found:
            parts[-1]['text'] += '\n' + line
    
    # strip
    for part in parts:
        part['text'] = part['text'].strip()
        # part['num_tokens'] = token_counter(f'{part["speaker"]}: {part["text"]}', model_name)

    # assert that no empty parts
    for part in parts:
        assert len(part['text']) > 0

    result['parts'] = parts

    return result

def load_debates(folder, download_if_missing=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # check if it contains at least 40 files and october-11-2012-the-biden-romney-vice-presidential-debate
    if len(os.listdir(folder)) < 40 or 'october-11-2012-the-biden-romney-vice-presidential-debate' not in os.listdir(folder):
        # clear folder
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        # sleep to be safe
        sleep(1)

        if download_if_missing:
            print('Downloading debates...')
            download_debates(folder)
        else:
            print(f'Debates not found in {folder} and download_if_missing=False, exiting.')
            raise FileNotFoundError
    
    debates = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r') as f:
            raw_debate = f.read()
        debate = parse_debate(raw_debate, debate_name=file, debate_index=len(debates))
        debates.append(debate)
    
    return debates

def slice_to_text(slice):
    text = ''
    for i, part in enumerate(slice['parts']):
        if not part['is_complete']:
            assert(i==0 or i==len(slice['parts'])-1)

        assert(part['speaker'] == part['speaker'].strip())
        assert(part['text'] == part['text'].strip())
        text += f"{part['speaker']}: "
        # if not part['is_complete'] and i == 0:
        if part['cut_left']:
            text += '... '
        text += f"{part['text']}"
        # if not part['is_complete'] and i == len(slice['parts']) - 1:
        if part['cut_right']:
            text += ' ...'
        text += '\n'
    
    return text[:-1]




def single_debate_to_slices_dataset(debate, slice_size=1000, slice_overlap_ratio=0.1, slice_cutoff_ratio=0.05, model_name='gpt-3.5-turbo-0613'):
    slices = []

    slice_overlap_size = int(slice_size * slice_overlap_ratio)
    slice_cutoff_size = int(slice_size * slice_cutoff_ratio)

    converation_tokens = []
    for i, part in enumerate(debate['parts']):
        speaker_text = f"{part['speaker']}: "
        if len(converation_tokens) > 0:
            speaker_text = '\n' + speaker_text
        for token in tokenizer(speaker_text, model_name):
            converation_tokens.append({
                'token': token,
                'part_index': i,
                'is_speaker': True
            })
        for token in tokenizer(part['text'], model_name):
            converation_tokens.append({
                'token': token,
                'part_index': i,
                'is_speaker': False
            })
    
    for i in range(0, len(converation_tokens), slice_size-slice_overlap_size):
        end = min(len(converation_tokens), i+slice_size)
        start = max(end-slice_size, 0)

        slice = {
            'id': f'{debate["id"]}-ss{slice_size}_so{slice_overlap_ratio}_co{slice_cutoff_ratio}_si{len(slices)}',
            'index': len(slices),
            'slice_size': slice_size,
            'slice_overlap_ratio': slice_overlap_ratio,
            'slice_cutoff_ratio': slice_cutoff_ratio,
            'start_token_index': start,
            'end_token_index': end,
            'debate_id': debate['id'],
            'parts': [{
                'part': debate['parts'][converation_tokens[start]['part_index']],
                'tokens': []
            }]
        }
        for token in converation_tokens[start:end]:
            # skip speaker tokens
            if token['is_speaker']:
                continue

            # check if part changed
            if debate['parts'][token['part_index']] != slice['parts'][-1]['part']:
                slice['parts'].append({
                    'part': debate['parts'][token['part_index']],
                    'tokens': []
                })
            
            slice['parts'][-1]['tokens'].append(token['token'])
        
        # for each part add meta data
        for i, part in enumerate(slice['parts']):
            part['num_tokens'] = len(part['tokens'])
            part['text'] = detokenizer(part['tokens'], model_name).strip()
            part['speaker'] = part['part']['speaker']
            part['part_id'] = part['part']['id']
            part['is_complete'] = len(part['text']) == len(part['part']['text'])
            if not part['is_complete']:
                assert(i==0 or i==len(slice['parts'])-1)
            part['cut_left'] = not part['part']['text'].startswith(part['text'])
            assert(not part['cut_left'] or i==0)
            part['cut_right'] = not part['part']['text'].endswith(part['text'])
            assert(not part['cut_right'] or i==len(slice['parts'])-1)
            if not (part['is_complete'] or (part['cut_right'] or part['cut_left'])):
                print('[WARNING] unclear cutoff')
                print(part['is_complete'], part['cut_right'],  part['cut_left'])
                print()
                print(part['text'])
                print()
                print(part['part']['text'])
            # assert(part['is_complete'] or (part['cut_right'] or part['cut_left']))

            del part['tokens']
        
        # remove parts with less than cutoff size if not complete
        if not slice['parts'][0]['is_complete']:
            if slice['parts'][0]['num_tokens'] < slice_cutoff_size:
                slice['parts'] = slice['parts'][1:]
        if not slice['parts'][-1]['is_complete']:
            if slice['parts'][-1]['num_tokens'] < slice_cutoff_size:
                slice['parts'] = slice['parts'][:-1]

        slice['num_tokens'] = token_counter(slice_to_text(slice), model_name)
        slice['num_parts'] = len(slice['parts'])
        slice['text'] = slice_to_text(slice)
        slice['speakers'] = list(sorted(set([part['speaker'] for part in slice['parts']])))
        slice['speakers_quantitative_contribution'] = {speaker: sum([part['num_tokens'] for part in slice['parts'] if part['speaker']==speaker]) for speaker in slice['speakers']}
        slice['speakers_quantitative_contribution_ratio'] = {speaker: slice['speakers_quantitative_contribution'][speaker]/slice['num_tokens'] for speaker in slice['speakers']}
        slice['speakers_num_parts'] = {speaker: len([part for part in slice['parts'] if part['speaker']==speaker]) for speaker in slice['speakers']}
        slice['speakers_num_parts_ratio'] = {speaker: slice['speakers_num_parts'][speaker]/slice['num_parts'] for speaker in slice['speakers']}
        
        assert(len(slice['parts']) > 0)
        # assert(sum([part['num_tokens'] for part in slice['parts']]) >= slice_size-slice_overlap_size*2)
    
        slices.append(slice)

    # check that all parts are in a slice
    assert(set([part['part_id'] for slice in slices for part in slice['parts']]) == set([part['id'] for part in debate['parts']]))

    # add defined_observables
    for slice in slices:
        slice_defined_observables = {}

        slice_id = slice['id']
        slice_index = slice['index']
        slice_size = slice['slice_size']
        slice_overlap_ratio = slice['slice_overlap_ratio']
        slice_cutoff_ratio = slice['slice_cutoff_ratio']
        slice_start_token_index = slice['start_token_index']
        slice_end_token_index = slice['end_token_index']
        slice_debate_id = slice['debate_id']
        slice_num_tokens = slice['num_tokens']
        slice_num_parts = slice['num_parts']
        slice_text = slice['text']
        slice_speakers = slice['speakers']
        slice_speakers_quantitative_contribution = slice['speakers_quantitative_contribution']
        slice_speakers_quantitative_contribution_ratio = slice['speakers_quantitative_contribution_ratio']
        slice_speakers_num_parts = slice['speakers_num_parts']

        # debate = debates_d[slice_debate_id]
        debate_date = debate['date']
        debate_possible_speakers = debate['possible_speakers']
        debate_name = debate['name']
        debate_index = debate['index']
        debate_id = debate['id']
        debate_parts = debate['parts']
        candidate_party_mapping = debate['candidate_party_mapping']

        debate_election_data = debate['election_data']
        debate_president_candidates = debate_election_data['president_candidates']
        debate_vice_president_candidates = debate_election_data['vice_president_candidates']
        debate_elected_president = debate_election_data['elected_president']
        debate_elected_vice_president = debate_election_data['elected_vice_president']
        debate_elected_party = candidate_party_mapping[debate_elected_president]
        debate_party_electoral_votes_mapping = debate_election_data['electoral_votes']
        debate_party_electoral_votes_ratio_mapping = {k:v/sum(debate_party_electoral_votes_mapping.values()) for k,v in debate_party_electoral_votes_mapping.items()}
        debate_party_popular_votes_mapping = debate_election_data['popular_votes']
        debate_party_popular_votes_ratio_mapping = {k:v/sum(debate_party_popular_votes_mapping.values()) for k,v in debate_party_popular_votes_mapping.items()}

        debate_observables = {
            'debate_year': int(debate_date['year']),
        }

        observables = {
            'slice_id': slice_id,
            'debate_id': slice_debate_id,
            'debate_total_electoral_votes': sum(debate_party_electoral_votes_mapping.values()),
            'debate_total_popular_votes': sum(debate_party_popular_votes_mapping.values()),
            'debate_elected_party': debate_elected_party,

            'slice_size': slice_size,

            **debate_observables,
        }

        for speaker in slice_speakers:
            speaker_party = candidate_party_mapping[speaker] if speaker in candidate_party_mapping else 'UNKNOWN'
            slice_defined_observables[speaker] = {
                **observables,

                'speaker': speaker,
                'speaker_party': speaker_party,
                'speaker_quantitative_contribution': slice_speakers_quantitative_contribution[speaker],
                'speaker_quantitative_contribution_ratio': slice_speakers_quantitative_contribution_ratio[speaker],
                'speaker_num_parts': slice_speakers_num_parts[speaker],
                'speaker_avg_part_size': slice_speakers_quantitative_contribution[speaker] / slice_speakers_num_parts[speaker],

                'speaker_electoral_votes': debate_party_electoral_votes_mapping[speaker_party] if speaker_party in debate_party_electoral_votes_mapping else 0,
                'speaker_electoral_votes_ratio': debate_party_electoral_votes_ratio_mapping[speaker_party] if speaker_party in debate_party_electoral_votes_ratio_mapping else 0,
                'speaker_popular_votes': debate_party_popular_votes_mapping[speaker_party] if speaker_party in debate_party_popular_votes_mapping else 0,
                'speaker_popular_votes_ratio': debate_party_popular_votes_ratio_mapping[speaker_party] if speaker_party in debate_party_popular_votes_ratio_mapping else 0,

                'speaker_won_election': int(speaker_party == debate_elected_party),
                'speaker_is_president_candidate': int(speaker in debate_president_candidates.values()),
                'speaker_is_vice_president_candidate': int(speaker in debate_vice_president_candidates.values()),
                'speaker_is_candidate': int(speaker in debate_president_candidates.values() or speaker in debate_vice_president_candidates.values()),
            }
    
        slice['defined_observables'] = slice_defined_observables

    return slices

def debates_to_slices_dataset(debates, slice_size=1000, slice_overlap_ratio=0.1, slice_cutoff_ratio=0.05, model_name='gpt-3.5-turbo-0613'):
    """
    converts debates to csv and tries has a max token size for a single slice and a min token size for the slice overlap
    we create slices and then mark for each seperate entry what slices it is in
    instead of only doing groups of slices we just cut off the slices and add the speaker to have easier and fairer preprocessing
    """

    slices = []
    for i, debate in enumerate(debates):
        new_slices = single_debate_to_slices_dataset(debate, slice_size=slice_size, slice_overlap_ratio=slice_overlap_ratio, slice_cutoff_ratio=slice_cutoff_ratio, model_name=model_name)
        
        slices += new_slices
    
    return slices


def load_slices_dataset(debates, slice_size, slice_overlap_ratio, slice_cutoff_ratio, slices_folder='cpd_slices_dataset', model_name='gpt-3.5-turbo-0613'):
    path = f'{slices_folder}/ss{slice_size}_sor{slice_overlap_ratio}_scr{slice_cutoff_ratio}.pickle'

    # create folder if not exists
    if not os.path.exists(slices_folder):
        os.makedirs(slices_folder)

    if os.path.exists(path):
        # load from pickle
        with open(path, 'rb') as f:
            slices = pickle.load(f)
    else:
        # load from debates
        slices = debates_to_slices_dataset(debates, slice_size=slice_size, slice_overlap_ratio=slice_overlap_ratio, slice_cutoff_ratio=slice_cutoff_ratio, model_name=model_name)
        # save to pickle
        with open(path, 'wb') as f:
            pickle.dump(slices, f)
    
    return slices

def load_paragraphs(slices):
    paragraphs = {}
    for slice in slices:
        for part in slice['parts']:
            if part['part']['id'] not in paragraphs:
                paragraphs[part['part']['id']] = part['part']
    return list(paragraphs.values())

def create_slices_text_files(slices, folder='cpd_debate_slices', slice_size=1000, slice_overlap_ratio=0.1, slice_cutoff_ratio=0.05):
    # for each part save the text to a file
    path = f'{folder}/texts_ss{slice_size}_sor{slice_overlap_ratio}_scr{slice_cutoff_ratio}.txt'
    if not os.path.exists(path):
        os.makedirs(path)
    for slice in slices:
        # with open(f'{path}/{slice["debate_id"]}_{slice["slice_index"]}', 'w') as f:
        with open(f'{path}/{slice["id"]}', 'w') as f:
            f.write(slice['text'])

# def main():
#     folder = 'cpd_slices_dataset'
#     slice_overlap_ratio=0.1
#     slice_cutoff_ratio=0.05

#     # # remove non empty folder
#     # if os.path.exists(folder):
#     #     shutil.rmtree(folder)

#     for slice_size, slice_overlap_ratio, slice_cutoff_ratio in [
#         (250, 0.1, 0.05),
#         (500, 0.1, 0.05),
#         (1000, 0.1, 0.05),
#         (2500, 0.1, 0.05),
#         (5000, 0.1, 0.05),
#         (12000, 0.1, 0.05),
#     ]:
#         # print(slice_size, int(slice_size*slice_cutoff_ratio))
#         s = load_slices_dataset(slice_size=slice_size, slice_overlap_ratio=slice_overlap_ratio, slice_cutoff_ratio=slice_cutoff_ratio, debates_folder='cpd_debates_raw', download_if_missing=True)
        
#         print(f'For ss={slice_size}, so={int(slice_overlap_ratio*slice_size)}, sc={int(slice_cutoff_ratio*slice_size)} we get {len(s)} slices')
        
#         create_slices_text_files(s, folder=folder, slice_size=slice_size, slice_overlap_ratio=slice_overlap_ratio, slice_cutoff_ratio=slice_cutoff_ratio)

# if __name__ == '__main__':
#     main()
