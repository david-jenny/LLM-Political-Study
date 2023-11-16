"""
This file contains classes creating the prompts and parsing the responses.
"""



import json



class SliceObservable:
    def __init__(self, datatype, detailed_name, name, description):
        self.datatype = datatype
        self.detailed_name = detailed_name
        self.name = name
        self.description = description

class SingleSliceObservablePrompt:
    def __init__(self, slice, observable):
        self.slice = slice
        self.observable = observable
        # self.response_prefix = f'{{\n\t"{self.observable.name}": '
        # self.response_prefix = ''
        # if self.observable.datatype == 'str':
        #     self.response_prefix += '"'

        # self.response_prefix = f'{{\n\t' # for some reason f'{{\n\t"{self.observable.name}": ' doesn't work and other variations don't work either or at least less reliably
        self.response_prefix = ''


        self.template="""You are a helpfull assistant tasked with completing information about part of a political debate. Here is the text you are working with:

---

{text}

---

All scores are between 0.0 and 1.0!
1.0 means that the quality of interest can't be stronger, 0.0 stands for a complete absence and 0.5 for how an average person in an average situation would be scored.
Strings are in ALL CAPS and without any additional information. If you are unsure about a string value, write 'UNCLEAR'.
Make sure that the response is a valid json object and that the keys are exactly as specified in the template!
Don't add any additional and unnecessary information or filler text!
Give your response as a json object with the following structure:

{{
\t"{name}": <{datatype} {description}>
}}

Now give your response as a complete, finished and correct json and don't write anything else:

"""

    def get_prompt(self):
        prompt = self.template.format(text=self.slice['text'], name=self.observable.name, datatype=self.observable.datatype, description=self.observable.description)

        return prompt + self.response_prefix
    
    def parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        try:
            decoded = json.loads(response)

            # check if the observable key is present
            if self.observable.name not in decoded:
                raise Exception(f'name {self.observable.name} not found in response!')
            
            # check if any other keys are present
            if len(decoded.keys()) > 1:
                raise Exception(f'Only {self.observable.name} is allowed as a key in the response, but found {list(decoded.keys())}!')

            # check if returned value is of the correct type
            value = decoded[self.observable.name]
            if self.observable.datatype == 'str':
                if not isinstance(value, str):
                    raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
            elif self.observable.datatype == 'float':
                if not isinstance(value, float):
                    raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
            
            # check if the value is in the correct range
            if self.observable.datatype == 'float':
                if value < 0 or value > 1:
                    raise Exception(f'Expected a float value between 0 and 1, but got {value}!')
            
            return { self.observable.detailed_name: value }
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e
    
    def parse_and_add_response(self, response, dataset):
        result = self.parse_response(response)

        slice_id = self.slice['id']
        dataset[slice_id] = dataset.get(slice_id, {})
        for key in result:
            if key in dataset[slice_id]:
                raise Exception(f'Key {key} already present in dataset:\n\t{dataset.keys()}')
            dataset[slice_id][key] = result[key]

class MultiSliceObservablePrompt:
    def __init__(self, slice, observables):
        self.slice = slice
        self.observables = observables

        self.response_prefix = ''

        # self.observable_template = """\"{name}\": <{datatype} {description}>"""
        self.observable_template = """{name}: <{datatype} {description}>""" # we keep bug, because we can't pay another 200$ for the fix

        self.template="""You are a helpfull assistant tasked with completing information about part of a political debate. Here is the text you are working with:

---

{text}

---

All scores are between 0.0 and 1.0!
1.0 means that the quality of interest can't be stronger, 0.0 stands for a complete absence and 0.5 for how an average person in an average situation would be scored.
Strings are in ALL CAPS and without any additional information. If you are unsure about a string value, write 'UNCLEAR'.
Make sure that the response is a valid json object and that the keys are exactly as specified in the template!
Don't add any additional and unnecessary information or filler text!
Give your response as a json object with the following structure:

{observables}

Now give your response as a complete, finished and correct json and don't write anything else:

"""

    def get_prompt(self):
        observables = ',\n\t'.join([
            self.observable_template.format(name=observable.name, datatype=observable.datatype, description=observable.description)
            for observable in self.observables
        ])
        observables = f'{{\n\t{observables}\n}}'

        prompt = self.template.format(text=self.slice['text'], observables=observables)

        return prompt + self.response_prefix


    def parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        results = {}

        try:
            decoded = json.loads(response)

            # check if the observable key is present
            for observable in self.observables:
                if observable.name not in decoded:
                    raise Exception(f'name {observable.name} not found in response!')
            
            # check if any other keys are present
            if len(decoded.keys()) > len(self.observables):
                raise Exception(f'Only {self.observables} is allowed as a key in the response, but found {list(decoded.keys())}!')

            # check if returned value is of the correct type
            for observable in self.observables:
                value = decoded[observable.name]
                if observable.datatype == 'str':
                    if not isinstance(value, str):
                        raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
                elif observable.datatype == 'float':
                    if not isinstance(value, float):
                        raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
                
                # check if the value is in the correct range
                if observable.datatype == 'float':
                    if value < 0 or value > 1:
                        raise Exception(f'Expected a float value between 0 and 1, but got {value}!')

                assert(observable.detailed_name not in results)
                results[observable.detailed_name] = value

            return results
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e

class SliceObservable:
    def __init__(self, datatype, detailed_name, name, description):
        self.datatype = datatype
        self.detailed_name = detailed_name
        self.name = name
        self.description = description

class SpeakerObservable:
    def __init__(self, datatype, detailed_name, name, description):
        self.datatype = datatype
        self.detailed_name = detailed_name
        self.name = name
        self.description = description

class MultiSpeakerObservablePrompt:
    def __init__(self, slice, speaker, observables):
        self.slice = slice
        self.speaker = speaker
        self.observables = observables

        self.response_prefix = ''

        # self.observable_template = """\"{name}\": <{datatype} {description}>"""
        self.observable_template = """{name}: <{datatype} {description}>""" # we keep bug, because we can't pay another 200$ for the fix

        self.template="""You are a helpfull assistant tasked with completing information about part of a political debate. Here is the text you are working with:

---

{text}

---

Your task is to complete information about the speaker {speaker} based on the text above.

All scores are between 0.0 and 1.0!
1.0 means that the quality of interest can't be stronger, 0.0 stands for a complete absence and 0.5 for how an average person in an average situation would be scored.
Strings are in ALL CAPS and without any additional information. If you are unsure about a string value, write 'UNCLEAR'.
Make sure that the response is a valid json object and that the keys are exactly as specified in the template!
Don't add any additional and unnecessary information or filler text!
Give your response as a json object with the following structure:

{observables}

Now give your response as a complete, finished and correct json and don't write anything else:

"""

    def get_prompt(self):
        observables = ',\n\t'.join([
            self.observable_template.format(name=observable.name, datatype=observable.datatype, description=observable.description)
            for observable in self.observables
        ])
        observables = f'{{\n\t{observables}\n}}'

        prompt = self.template.format(text=self.slice['text'], speaker=self.speaker, observables=observables)

        return prompt + self.response_prefix


    def parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        results = {}

        try:
            decoded = json.loads(response)

            # if pro_democratic instead of pro democratic in decoded then fix it
            observable_names = [observable.name for observable in self.observables]
            for key in list(decoded.keys()):
                if key not in observable_names:
                    for replacement in [('_', ' '), (' ', '_')]:
                        r = key.replace(*replacement)
                        if r != key and r in observable_names:
                            decoded[r] = decoded[key]
                            del decoded[key]
                            break

            # check if the observable key is present
            for observable in self.observables:
                if observable.name not in decoded:
                    raise Exception(f'name {observable.name} not found in response!')
            
            # check if any other keys are present
            if len(decoded.keys()) > len(self.observables):
                raise Exception(f'Only {self.observables} is allowed as a key in the response, but found {list(decoded.keys())}!')

            # check if returned value is of the correct type
            for observable in self.observables:
                value = decoded[observable.name]
                if observable.datatype == 'str':
                    if not isinstance(value, str):
                        raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
                elif observable.datatype == 'float':
                    if not isinstance(value, float):
                        raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
                
                # check if the value is in the correct range
                if observable.datatype == 'float':
                    if value < 0 or value > 1:
                        raise Exception(f'Expected a float value between 0 and 1, but got {value}!')

                assert(observable.detailed_name not in results)
                results[observable.detailed_name] = value

            return {self.speaker: results}
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e

    def parse_and_add_response(self, response, dataset):
        results = self.parse_response(response)

        slice_id = self.slice['id']
        dataset[slice_id] = dataset.get(slice_id, {})
        for speaker, values in results.items():
            dataset[slice_id][speaker] = dataset[slice_id].get(speaker, {})
            for observable, value in values.items():
                assert(observable not in dataset[slice_id][speaker])
                dataset[slice_id][speaker][observable] = value

class MultiSpeakerObservableMultiSpeakersPrompt:
    name = 'MultiSpeakerObservableMultiSpeakersPrompt'
    short_name = 'msomp'

    def __init__(self, slice, speakers, observables):
        self.slice = slice
        self.speakers = speakers
        self.observables = observables

        self.response_prefix = ''

        self.observable_template = """\"{name}\": <{datatype} {description}>"""

        self.template="""You are a helpfull assistant tasked with completing information about part of a political debate. Here is the text you are working with:

---

{text}

---

Your task is to complete information about the speakers based on the text above.

Here are the speakers:
{speakers}
Don't leave any out or add additional ones!

All scores are between 0.0 and 1.0!
1.0 means that the quality of interest can't be stronger, 0.0 stands for a complete absence and 0.5 for how an average person in an average situation would be scored.
Strings are in ALL CAPS and without any additional information. If you are unsure about a string value, write 'UNCLEAR'.
Make sure that the response is a valid json object and that the keys are exactly as specified in the template!
Don't add any additional and unnecessary information or filler text!
Give your response as a json object with the following structure:

{{
\t<str speaker>: {observables},
\t...
}}

Now give your response as a complete, finished and correct json including each speaker and don't write anything else:

"""

    def get_prompt(self):
        observables = ',\n\t\t'.join([
            self.observable_template.format(name=observable.name, datatype=observable.datatype, description=observable.description)
            for observable in self.observables
        ])
        observables = f'{{\n\t\t{observables}\n\t}}'

        prompt = self.template.format(text=self.slice['text'], speakers=self.speakers, observables=observables)

        return prompt + self.response_prefix


    def parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        results = {}

        try:
            decoded = json.loads(response)

            # check if all speakers are present
            for speaker in self.speakers:
                if speaker not in decoded:
                    raise Exception(f'speaker {speaker} not found in response!')
            assert(len(decoded.keys()) == len(self.speakers))

            for speaker, values in decoded.items():
                assert(speaker not in results)
                results[speaker] = {}

                # check if the observable key is present
                for observable in self.observables:
                    if observable.name not in values:
                        raise Exception(f'name {observable.name} not found in response!')
                
                # check if any other keys are present
                if len(values.keys()) > len(self.observables):
                    raise Exception(f'Only {self.observables} is allowed as a key in the response, but found {list(decoded.keys())}!')

                # check if returned value is of the correct type
                for observable in self.observables:
                    value = values[observable.name]
                    if observable.datatype == 'str':
                        if not isinstance(value, str):
                            raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
                    elif observable.datatype == 'float':
                        if not isinstance(value, float):
                            raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
                    
                    # check if the value is in the correct range
                    if observable.datatype == 'float':
                        if value < 0 or value > 1:
                            raise Exception(f'Expected a float value between 0 and 1, but got {value}!')

                    assert(observable.detailed_name not in results[speaker])
                    results[speaker][observable.detailed_name] = value

            return results
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e

    def parse_and_add_response(self, response, dataset):
        results = self.parse_response(response)

        slice_id = self.slice['id']
        dataset[slice_id] = dataset.get(slice_id, {})
        for speaker, values in results.items():
            dataset[slice_id][speaker] = dataset[slice_id].get(speaker, {})
            for observable, value in values.items():
                assert(observable not in dataset[slice_id][speaker])
                dataset[slice_id][speaker][observable] = value
