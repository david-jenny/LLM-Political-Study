"""
This file contains classes creating the prompts and parsing the responses.
"""



import json



class SliceVariable:
    def __init__(self, datatype, detailed_name, name, description):
        self.datatype = datatype
        self.detailed_name = detailed_name
        self.name = name
        self.description = description

class SingleSliceVariablePrompt:
    name = 'SingleSliceVariable_Prompt'
    short_name = 'SSLV'

    def __init__(self, slice, variable):
        self.slice = slice
        self.variable = variable
        # self.response_prefix = f'{{\n\t"{self.variable.name}": '
        # self.response_prefix = ''
        # if self.variable.datatype == 'str':
        #     self.response_prefix += '"'

        # self.response_prefix = f'{{\n\t' # for some reason f'{{\n\t"{self.variable.name}": ' doesn't work and other variations don't work either or at least less reliably
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
        prompt = self.template.format(text=self.slice['text'], name=self.variable.name, datatype=self.variable.datatype, description=self.variable.description)

        return prompt + self.response_prefix
    
    def _parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        try:
            decoded = json.loads(response)

            # check if the variable key is present
            if self.variable.name not in decoded:
                raise Exception(f'name {self.variable.name} not found in response!')
            
            # check if any other keys are present
            if len(decoded.keys()) > 1:
                raise Exception(f'Only {self.variable.name} is allowed as a key in the response, but found {list(decoded.keys())}!')

            # check if returned value is of the correct type
            value = decoded[self.variable.name]
            if self.variable.datatype == 'str':
                if not isinstance(value, str):
                    raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
            elif self.variable.datatype == 'float':
                if not isinstance(value, float):
                    raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
            
            # check if the value is in the correct range
            if self.variable.datatype == 'float':
                if value < 0 or value > 1:
                    raise Exception(f'Expected a float value between 0 and 1, but got {value}!')
            
            return { self.variable.detailed_name: value }
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e
    
    def parse_response(self, response):
        values = self._parse_response(response)
        results = []

        assert(len(values) == 1)

        for variable, value in values.items():
            assert(variable == self.variable.detailed_name)
            obs = self.variable
            
            results.append({
                'slice_id': self.slice['id'],
                'prompt_type': self.name,

                'datatype': obs.datatype,
                'detailed_name': obs.detailed_name,
                'name': obs.name,
                'description': obs.description,

                'value': value
            })
        
        return results
    
    def parse_and_add_response(self, response, dataset):
        result = self._parse_response(response)

        slice_id = self.slice['id']
        dataset[slice_id] = dataset.get(slice_id, {})
        for key in result:
            if key in dataset[slice_id]:
                raise Exception(f'Key {key} already present in dataset:\n\t{dataset.keys()}')
            dataset[slice_id][key] = result[key]

class MultiSliceVariablePrompt:
    name = 'MultiSliceVariables_Prompt'
    short_name = 'MSLV'

    def __init__(self, slice, variables):
        self.slice = slice
        self.variables = variables

        self.response_prefix = ''

        # self.variable_template = """\"{name}\": <{datatype} {description}>"""
        self.variable_template = """{name}: <{datatype} {description}>""" # we keep bug, because we can't pay another 200$ for the fix

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

{variables}

Now give your response as a complete, finished and correct json and don't write anything else:

"""

    def get_prompt(self):
        variables = ',\n\t'.join([
            self.variable_template.format(name=variable.name, datatype=variable.datatype, description=variable.description)
            for variable in self.variables
        ])
        variables = f'{{\n\t{variables}\n}}'

        prompt = self.template.format(text=self.slice['text'], variables=variables)

        return prompt + self.response_prefix


    def _parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        results = {}

        try:
            decoded = json.loads(response)

            # check if the variable key is present
            for variable in self.variables:
                if variable.name not in decoded:
                    raise Exception(f'name {variable.name} not found in response!')
            
            # check if any other keys are present
            if len(decoded.keys()) > len(self.variables):
                raise Exception(f'Only {self.variables} is allowed as a key in the response, but found {list(decoded.keys())}!')

            # check if returned value is of the correct type
            for variable in self.variables:
                value = decoded[variable.name]
                if variable.datatype == 'str':
                    if not isinstance(value, str):
                        raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
                elif variable.datatype == 'float':
                    if not isinstance(value, float):
                        raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
                
                # check if the value is in the correct range
                if variable.datatype == 'float':
                    if value < 0 or value > 1:
                        raise Exception(f'Expected a float value between 0 and 1, but got {value}!')

                assert(variable.detailed_name not in results)
                results[variable.detailed_name] = value

            return results
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e
        
    def parse_response(self, response):
        values = self._parse_response(response)
        results = []

        vars_lookup = {v.detailed_name: v for v in self.variables}

        for variable, value in values.items():
            obs = vars_lookup[variable]
            
            results.append({
                'slice_id': self.slice['id'],
                'prompt_type': self.name,

                'datatype': obs.datatype,
                'detailed_name': obs.detailed_name,
                'name': obs.name,
                'description': obs.description,

                'value': value
            })
        
        return results

class SliceVariable:
    def __init__(self, datatype, detailed_name, name, description):
        self.datatype = datatype
        self.detailed_name = detailed_name
        self.name = name
        self.description = description

class SpeakerVariable:
    def __init__(self, datatype, detailed_name, name, description):
        self.datatype = datatype
        self.detailed_name = detailed_name
        self.name = name
        self.description = description

class MultiSpeakerVariableSingleSpeakerPrompt:
    name = 'MultiSpeakerVariables_SingleSpeaker_Prompt'
    short_name = 'SS_MSV'

    def __init__(self, slice, speaker, variables):
        self.slice = slice
        self.speaker = speaker
        self.variables = variables

        self.response_prefix = ''

        # self.variable_template = """\"{name}\": <{datatype} {description}>"""
        self.variable_template = """{name}: <{datatype} {description}>""" # we keep bug, because we can't pay another 200$ for the fix

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

{variables}

Now give your response as a complete, finished and correct json and don't write anything else:

"""

    def get_prompt(self):
        variables = ',\n\t'.join([
            self.variable_template.format(name=variable.name, datatype=variable.datatype, description=variable.description)
            for variable in self.variables
        ])
        variables = f'{{\n\t{variables}\n}}'

        prompt = self.template.format(text=self.slice['text'], speaker=self.speaker, variables=variables)

        return prompt + self.response_prefix


    def _parse_response(self, response):
        response = response.strip()
        response = self.response_prefix + response

        results = {}

        try:
            decoded = json.loads(response)

            # if pro_democratic instead of pro democratic in decoded then fix it
            variable_names = [variable.name for variable in self.variables]
            for key in list(decoded.keys()):
                if key not in variable_names:
                    for replacement in [('_', ' '), (' ', '_')]:
                        r = key.replace(*replacement)
                        if r != key and r in variable_names:
                            decoded[r] = decoded[key]
                            del decoded[key]
                            break

            # check if the variable key is present
            for variable in self.variables:
                if variable.name not in decoded:
                    raise Exception(f'name {variable.name} not found in response!')
            
            # check if any other keys are present
            if len(decoded.keys()) > len(self.variables):
                raise Exception(f'Only {self.variables} is allowed as a key in the response, but found {list(decoded.keys())}!')

            # check if returned value is of the correct type
            for variable in self.variables:
                value = decoded[variable.name]
                if variable.datatype == 'str':
                    if not isinstance(value, str):
                        raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
                elif variable.datatype == 'float':
                    if not isinstance(value, float):
                        raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
                
                # check if the value is in the correct range
                if variable.datatype == 'float':
                    if value < 0 or value > 1:
                        raise Exception(f'Expected a float value between 0 and 1, but got {value}!')

                assert(variable.detailed_name not in results)
                results[variable.detailed_name] = value

            return {self.speaker: results}
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e
        
    def parse_response(self, response):
        values = self._parse_response(response)
        results = []
        
        vars_lookup = {v.detailed_name: v for v in self.variables}

        values = values[self.speaker]
        for variable, value in values.items():
            obs = vars_lookup[variable]
            
            results.append({
                'slice_id': self.slice['id'],
                'prompt_type': self.name,

                'speaker': self.speaker,
                'datatype': obs.datatype,
                'detailed_name': obs.detailed_name,
                'name': obs.name,
                'description': obs.description,

                'value': value
            })
        
        return results

    def parse_and_add_response(self, response, dataset):
        results = self._parse_response(response)

        slice_id = self.slice['id']
        dataset[slice_id] = dataset.get(slice_id, {})
        for speaker, values in results.items():
            dataset[slice_id][speaker] = dataset[slice_id].get(speaker, {})
            for variable, value in values.items():
                assert(variable not in dataset[slice_id][speaker])
                dataset[slice_id][speaker][variable] = value


class PerturbedSingleSpeakerVariableSingleSpeakerPrompt:
    name = 'Perturbed_SpeakerVariable_SingleSpeaker_Prompt'
    short_name = 'P_SS_SSV'

    def __init__(self, slice, speaker, variable, given_variable, real_given_value, pertubation, original_output_value):
        self.slice = slice
        self.speaker = speaker
        self.variable = variable
        self.variables = [variable, given_variable]
        self.given_variable = given_variable
        self.real_given_value = real_given_value
        self.pertubation = pertubation
        self.original_output_value = original_output_value


        self.response_prefix = ''

        # self.variable_template = """\"{name}\": <{datatype} {description}>"""
        self.variable_template = """\"{name}\": <{datatype} {description}>""" # we keep bug, because we can't pay another 200$ for the fix

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

{variables}

Now give your response as a complete, finished and correct json and don't write anything else:

"""

    def get_prompt(self):
        variables = ',\n\t'.join([
            self.variable_template.format(name=variable.name, datatype=variable.datatype, description=variable.description)
            for variable in self.variables
        ])
        variables = f'{{\n\t{variables}\n}}'

        prompt = self.template.format(text=self.slice['text'], speaker=self.speaker, variables=variables)

        perturbed_value = self.real_given_value + self.pertubation
        partial_fixed_response = f'{{\n\t"{self.given_variable.name}": {perturbed_value},\n\t"{self.variable.name}": '

        return prompt + partial_fixed_response + self.response_prefix
        
    def parse_response(self, response):
        try:
            response = response.strip()
            if response.endswith('}'):
                response = response[:-1].strip()
            result = float(response)
        except Exception as e:
            print(f'[ERROR] Failed to parse response which should just be a float: {response}')
            raise e


        return [{
            'slice_id': self.slice['id'],
            'prompt_type': self.name,

            'speaker': self.speaker,

            'datatype': self.variable.datatype,
            'detailed_name': self.variable.detailed_name,
            'name': self.variable.name,
            'description': self.variable.description,

            'given_name': self.given_variable.name,
            'given_detailed_name': self.given_variable.detailed_name,
            'real_value': self.real_given_value,
            'pertubation': self.pertubation,
            'perturbed_value': self.real_given_value + self.pertubation,
            'original_output_value': self.original_output_value,

            'value': result,
        }]

class SingleSpeakerVariableSingleSpeakerPrompt(MultiSpeakerVariableSingleSpeakerPrompt):
    name = 'SingleSpeakerVariable_SingleSpeaker_Prompt'
    short_name = 'SS_SSV'

    def __init__(self, slice, speaker, variable):
        super().__init__(slice, speaker, [variable])

class MultiSpeakerVariablesMultiSpeakersPrompt:
    name = 'MultiSpeakerVariables_MultiSpeakers_Prompt'
    short_name = 'MS_MSV'

    def __init__(self, slice, speakers, variables):
        self.slice = slice
        self.speakers = speakers
        self.variables = variables

        self.response_prefix = ''

        self.variable_template = """\"{name}\": <{datatype} {description}>"""

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
\t<str speaker>: {variables},
\t...
}}

Now give your response as a complete, finished and correct json including each speaker and don't write anything else:

"""

    def get_prompt(self):
        variables = ',\n\t\t'.join([
            self.variable_template.format(name=variable.name, datatype=variable.datatype, description=variable.description)
            for variable in self.variables
        ])
        variables = f'{{\n\t\t{variables}\n\t}}'

        prompt = self.template.format(text=self.slice['text'], speakers=self.speakers, variables=variables)

        return prompt + self.response_prefix


    def _parse_response(self, response):
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

                # check if the variable key is present
                for variable in self.variables:
                    if variable.name not in values:
                        raise Exception(f'name {variable.name} not found in response!')
                
                # check if any other keys are present
                if len(values.keys()) > len(self.variables):
                    raise Exception(f'Only {self.variables} is allowed as a key in the response, but found {list(decoded.keys())}!')

                # check if returned value is of the correct type
                for variable in self.variables:
                    value = values[variable.name]
                    if variable.datatype == 'str':
                        if not isinstance(value, str):
                            raise Exception(f'Expected a string value, but got {value} of type {type(value)}!')
                    elif variable.datatype == 'float':
                        if not isinstance(value, float):
                            raise Exception(f'Expected a float value, but got {value} of type {type(value)}!')
                    
                    # check if the value is in the correct range
                    if variable.datatype == 'float':
                        if value < 0 or value > 1:
                            raise Exception(f'Expected a float value between 0 and 1, but got {value}!')

                    assert(variable.detailed_name not in results[speaker])
                    results[speaker][variable.detailed_name] = value

            return results
        except Exception as e:
            print(f'[ERROR] Failed to parse response: {e}')
            raise e
    
    def parse_response(self, response):
        values = self._parse_response(response)
        results = []

        vars_lookup = {v.detailed_name: v for v in self.variables}

        for speaker, tmp in values.items():
            for variable, value in tmp.items():
                obs = vars_lookup[variable]
                
                results.append({
                    'slice_id': self.slice['id'],
                    'prompt_type': self.name,

                    'speaker': speaker,
                    'datatype': obs.datatype,
                    'detailed_name': obs.detailed_name,
                    'name': obs.name,
                    'description': obs.description,

                    'value': value
                })
        
        return results

    def parse_and_add_response(self, response, dataset):
        results = self._parse_response(response)

        slice_id = self.slice['id']
        dataset[slice_id] = dataset.get(slice_id, {})
        for speaker, values in results.items():
            dataset[slice_id][speaker] = dataset[slice_id].get(speaker, {})
            for variable, value in values.items():
                assert(variable not in dataset[slice_id][speaker])
                dataset[slice_id][speaker][variable] = value
