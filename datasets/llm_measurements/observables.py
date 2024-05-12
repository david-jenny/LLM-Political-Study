from .prompt_builder import SliceVariable, SingleSliceVariablePrompt, MultiSliceVariablePrompt, SpeakerVariable, MultiSpeakerVariableSingleSpeakerPrompt, MultiSpeakerVariablesMultiSpeakersPrompt


contextual_variables_descriptions = {
    'slice_id': 'unique identifier for a slice',
    'debate_id': 'unique identifier for debate',
    'slice_size': 'the target token size of the slice',
    'debate_year': 'the year in which the debate took place',

    'debate_total_electoral_votes': 'total electoral votes in election',
    'debate_total_popular_votes': 'total popular votes in election',

    'debate_elected_party': 'party that was elected after debates',

    'speaker': 'the name of the speaker that is examined in the context of the current slice',
    'speaker_party': 'party of the speaker',
    'speaker_quantitative_contribution': 'quantitative contribution in tokens of the speaker to this slice',
    'speaker_quantitative_contribution_ratio': 'ratio of contribution of speaker to everything that was said',
    'speaker_num_parts': 'number of paragraphs the speaker has in current slice',
    'speaker_avg_part_size': 'average size of paragraph for speaker',
    'speaker_electoral_votes': 'electoral votes that the candidates party scored',
    'speaker_electoral_votes_ratio': 'ratio of electoral votes that the candidates party scored',
    'speaker_popular_votes': 'popular votes that the candidates party scored',
    'speaker_popular_votes_ratio': 'ratio of popular votes that the candidates party scored',
    'speaker_won_election': 'flag (0 or 1) that says if speakers party won the election',
    'speaker_is_president_candidate': 'flag (0 or 1) that says whether the speaker is a presidential candidate',
    'speaker_is_vice_president_candidate': 'flag (0 or 1) that says whether the speaker is a vice presidential candidate',
    'speaker_is_candidate': 'flag (0 or 1) that says whether the speaker is a presidential or vice presidential candidate',
}

slice_variables = {
    'content quality (filler)': SliceVariable(
        'float', 'content quality (filler)', 'content quality',
        'Is there any content in this part of the debate or is it mostly filler?',
    ),
    'content quality (speaker)': SliceVariable(
        'float', 'content quality (speaker)', 'content quality',
        'Is there any valuable content in this part of the debate that can be used for further analysis of how well the speakers can argue their points?',
    ),
    'content quality (dataset)': SliceVariable(
        'float', 'content quality (dataset)', 'content quality',
        'We want to create a dataset to study how well the speakers can argue, convery information and what leads to winning an election. Should this part of the debate be included in the dataset?',
    ),
    'topic predictiveness (usefullness)': SliceVariable(
        'float', 'topic predictiveness (usefullness)', 'topic predictiveness',
        'Can this part of the debate be used to predict the topic of the debate?',
    ),
    'topic (max3)': SliceVariable(
        'str', 'topic (max3)', 'topic',
        'Which topic is being discussed in this part of the debate? Respond with a short, compact and general title with max 3 words in all caps.'
    ),
}

slice_variables_groups = {}
for variable in slice_variables.values():
    slice_variables_groups[variable.name] = slice_variables_groups.get(variable.name, []) + [variable]

predictor_variables = {
    'egotistical (benefit)': SpeakerVariable(
        'float', 'egotistical (benefit)', 'egotistical',
        'How much do the speaker\'s arguments benefit the speaker himself?',
    ),

    'persuasiveness (convincing)': SpeakerVariable(
        'float', 'persuasiveness (convincing)', 'persuasiveness',
        'How convincing are the arguments or points made by the speaker?',
    ),

    'clarity (understandable)': SpeakerVariable(
        'float', 'clarity (understandable)', 'clarity',
        'How clear and understandable is the speaker\'s arguments?',
    ),
    'clarity (easiness)': SpeakerVariable(
        'float', 'clarity (easiness)', 'clarity',
        'How easy are the speaker\'s arguments to understand for a general audience?',
    ),
    'clarity (clarity)': SpeakerVariable(
        'float', 'clarity (clarity)', 'clarity',
        'Is the speaker able to convey their arguments in a clear and comprehensible manner?',
    ),


    'contribution (quality)': SpeakerVariable(
        'float', 'contribution (quality)', 'contribution',
        'How good is the speaker\'s contribution to the discussion?',
    ),
    'contribution (quantity)': SpeakerVariable(
        'float', 'contribution (quantity)', 'contribution',
        'How much does the speaker contribute to the discussion?',
    ),

    'truthfulness (thruthullness)': SpeakerVariable(
        'float', 'truthfulness (thruthullness)', 'truthfulness',
        'How truthful are the speaker\'s arguments?',
    ),
    # 'truthfulness (lie)': SpeakerObservable(
    #     'float', 'truthfulness (lie)', 'truthfulness',
    #     'How much does the speaker lie?',
    # ), # always gave 0.5 and was not what we wanted...

    'bias (bias)': SpeakerVariable(
        'float', 'bias (bias)', 'bias',
        'How biased is the speaker?',
    ),

    'manipulation (manipulation)': SpeakerVariable(
        'float', 'manipulation (manipulation)', 'manipulation',
        'Is the speaker trying to subtly guide the reader towards a particular conclusion or opinion?',
    ),
    'manipulation (underhanded)': SpeakerVariable(
        'float', 'manipulation (underhanded)', 'manipulation',
        'Is the speaker trying to underhandedly guide the reader towards a particular conclusion or opinion?',
    ),

    'evasiveness (avoid)': SpeakerVariable(
        'float', 'evasiveness (avoid)', 'evasiveness',
        'Does the speaker avoid answering questions or addressing certain topics?',
    ),
    'evasiveness (ignore)': SpeakerVariable(
        'float', 'evasiveness (ignore)', 'evasiveness',
        'Does the speaker ignore certain topics or questions?',
    ),
    'evasiveness (dodge)': SpeakerVariable(
        'float', 'evasiveness (dodge)', 'evasiveness',
        'Does the speaker dodge certain topics or questions?',
    ),
    'evasiveness (evade)': SpeakerVariable(
        'float', 'evasiveness (evade)', 'evasiveness',
        'Does the speaker evade certain topics or questions?',
    ),

    'relevant (relevant)': SpeakerVariable(
        'float', 'relevant (relevant)', 'relevant',
        'How relevant is the speaker\'s arguments to the stated topic or subject?',
    ),

    'everyday relevance (relevance)': SpeakerVariable(
        'float', 'relevance (relevance)', 'relevance',
        'Do the speaker’s arguments and issues addressed have relevance to the everyday lives of the audience?',
    ),

    'conciseness (efficiency)': SpeakerVariable(
        'float', 'conciseness (efficiency)', 'conciseness',
        'Does the speaker express his points efficiently without unnecessary verbiage?',
    ),
    'conciseness (concise)': SpeakerVariable(
        'float', 'conciseness (concise)', 'conciseness',
        'Does the speaker express his points concisely?',
    ),

    # 'depth of analysis (depth)': SpeakerObservable(
    #     'float', 'depth of analysis (depth)', 'depth of analysis',
    #     'Does the speaker offer deep, thoughtful insights or is it more superficial?',
    # ),
    # 'depth of analysis (insight)': SpeakerObservable(
    #     'float', 'depth of analysis (insight)', 'depth of analysis',
    #     'Does the speaker offer insightful analysis?',
    # ), # not very usefull and always gave same result...

    'use of evidence (evidence)': SpeakerVariable(
        'float', 'use of evidence (evidence)', 'use of evidence',
        'Does the speaker use solid evidence to support his points?',
    ),

    'emotional appeal (emotional)': SpeakerVariable(
        'float', 'emotional appeal (emotional)', 'emotional appeal',
        'Does the speaker use emotional language or appeals to sway the reader?',
    ),

    'objectivity (unbiased)': SpeakerVariable(
        'float', 'objectivity (unbiased)', 'objectivity',
        'Does the speaker attempt to present an unbiased, objective view of the topic?',
    ),

    'sensationalism (exaggerated)': SpeakerVariable(
        'float', 'sensationalism (exaggerated)', 'sensationalism',
        'Does the speaker use exaggerated or sensational language to attract attention?',
    ),

    'controversiality (controversial)': SpeakerVariable(
        'float', 'controversiality (controversial)', 'controversiality',
        'Does the speaker touch on controversial topics or take controversial stances?',
    ),

    'coherence (coherent)': SpeakerVariable(
        'float', 'coherence (coherent)', 'coherence',
        'Do the speaker\'s points logically follow from one another?',
    ),

    'consistency (consistent)': SpeakerVariable(
        'float', 'consistency (consistent)', 'consistency',
        'Are the arguments and viewpoints the speaker presents consistent with each other?',
    ),

    # 'originality (original)': SpeakerObservable(
    #     'float', 'originality (original)', 'originality',
    #     'Does the speaker present new ideas or perspectives, or does he rehash existing ones?',
    # ), # always gave 0.5

    'factuality (factual)': SpeakerVariable(
        'float', 'factuality (factual)', 'factuality',
        'How much of the speaker\'s arguments are based on factual information versus opinion?',
    ),

    'completeness (complete)': SpeakerVariable(
        'float', 'completeness (complete)', 'completeness',
        'Does the speaker cover the topic fully and address all relevant aspects?',
    ),

    'quality of sources (reliable)': SpeakerVariable(
        'float', 'quality of sources (reliable)', 'quality of sources',
        'How reliable and credible are the sources used by the speaker?',
    ),

    'balance (balanced)': SpeakerVariable(
        'float', 'balance (balanced)', 'balance',
        'Does the speaker present multiple sides of the issue, or is it one-sided?',
    ),

    'tone is professional (tone)': SpeakerVariable(
        'float', 'tone is professional (tone)', 'tone is professional',
        'Does the speaker use a professional tone?',
    ),
    'tone is conversational (tone)': SpeakerVariable(
        'float', 'tone is conversational (tone)', 'tone is conversational',
        'Does the speaker use a conversational tone?',
    ),
    'tone is academic (tone)': SpeakerVariable(
        'float', 'tone is academic (tone)', 'tone is academic',
        'Does the speaker use an academic tone?',
    ),

    'accessibility (accessibility)': SpeakerVariable(
        'float', 'accessibility (accessibility)', 'accessibility',
        'How easily can the speaker be understood by a general audience?',
    ),

    'engagement (engagement)': SpeakerVariable(
        'float', 'engagement (engagement)', 'engagement',
        'How much does the speaker draw in and hold the reader\'s attention?',
    ),
    # 'engagement (participation)': SpeakerVariable(
    #     'float', 'engagement (participation)', 'engagement',
    #     'Does the speaker actively engage the audience, encouraging participation and dialogue?',
    # ),

    'adherence to rules (adherence)': SpeakerVariable(
        'float', 'adherence to rules (adherence)', 'adherence to rules',
        'Does the speaker respect and adhere to the rules and format of the debate or discussion?',
    ),

    'respectfulness (respectfulness)': SpeakerVariable(
        'float', 'respectfulness (respectfulness)', 'respectfulness',
        'Does the speaker show respect to others involved in the discussion, including the moderator and other participants?',
    ),

    'interruptions (interruptions)': SpeakerVariable(
        'float', 'interruptions (interruptions)', 'interruptions',
        'How often does the speaker interrupt others when they are speaking?',
    ),

    'time management (time management)': SpeakerVariable(
        'float', 'time management (time management)', 'time management',
        'Does the speaker make effective use of their allotted time, and respect the time limits set for their responses?',
    ),

    'responsiveness (responsiveness)': SpeakerVariable(
        'float', 'responsiveness (responsiveness)', 'responsiveness',
        'How directly does the speaker respond to questions or prompts from the moderator or other participants?',
    ),

    'decorum (decorum)': SpeakerVariable(
        'float', 'decorum (decorum)', 'decorum',
        'Does the speaker maintain the level of decorum expected in the context of the discussion?',
    ),

    'venue respect (venue respect)': SpeakerVariable(
        'float', 'venue respect (venue respect)', 'venue respect',
        'Does the speaker show respect for the venue and event where the debate is held?',
    ),

    'language appropriateness (language appropriateness)': SpeakerVariable(
        'float', 'language appropriateness (language appropriateness)', 'language appropriateness',
        'Does the speaker use language that is appropriate for the setting and audience?',
    ),

    # 'nonverbal communication (nonverbal communication)': SpeakerObservable(
    #     'float', 'nonverbal communication (nonverbal communication)', 'nonverbal communication',
    #     'How much does the speaker use nonverbal communication to enhance their message?',
    # ),

    # 'body language (nonverbal communication)': SpeakerObservable(
    #     'float', 'body language (nonverbal communication)', 'body language',
    #     'How much does the speaker use body language to enhance their message?',
    # ),

    'contextual awareness (contextual awareness)': SpeakerVariable(
        'float', 'contextual awareness (contextual awareness)', 'contextual awareness',
        'How much does the speaker demonstrate awareness of the context of the discussion?',
    ),

    'confidence (confidence)': SpeakerVariable(
        'float', 'confidence (confidence)', 'confidence',
        'How confident does the speaker appear?',
    ),

    'fair play (fair play)': SpeakerVariable(
        'float', 'fair play (fair play)', 'fair play',
        'Does the speaker engage in fair debating tactics, or do they resort to logical fallacies, personal attacks, or other unfair tactics?',
    ),

    'listening skills (listening skills)': SpeakerVariable(
        'float', 'listening skills (listening skills)', 'listening skills',
        'Does the speaker show that they are actively listening and responding to the points made by others?',
    ),

    'civil discourse (civil discourse)': SpeakerVariable(
        'float', 'civil discourse (civil discourse)', 'civil discourse',
        'Does the speaker contribute to maintaining a climate of civil discourse, where all participants feel respected and heard?',
    ),

    'respect for diverse opinions (respect for diverse opinions)': SpeakerVariable(
        'float', 'respect for diverse opinions (respect for diverse opinions)', 'respect for diverse opinions',
        'Does the speaker show respect for viewpoints different from their own, even while arguing against them?',
    ),

    'preparation (preparation)': SpeakerVariable(
        'float', 'preparation (preparation)', 'preparation',
        'Does the speaker seem well-prepared for the debate, demonstrating a good understanding of the topics and questions at hand?',
    ),

    'resonance (resonance)': SpeakerVariable(
        'float', 'resonance (resonance)', 'resonance',
        'Does the speaker’s message resonate with the audience, aligning with their values, experiences, and emotions?',
    ),

    'authenticity (authenticity)': SpeakerVariable(
        'float', 'authenticity (authenticity)', 'authenticity',
        'Does the speaker come across as genuine and authentic in their communication and representation of issues?',
    ),

    'empathy (empathy)': SpeakerVariable(
        'float', 'empathy (empathy)', 'empathy',
        'Does the speaker demonstrate empathy and understanding towards the concerns and needs of the audience?',
    ),

    'innovation (innovation)': SpeakerVariable(
        'float', 'innovation (innovation)', 'innovation',
        'Does the speaker introduce innovative ideas and perspectives that contribute to the discourse?',
    ),

    'outreach (penetration)': SpeakerVariable(
        'float', 'outreach (penetration)', 'outreach US',
        'How effectively do the speaker’s arguments penetrate various demographics and social groups within the US society?',
    ),
    'outreach (relatability)': SpeakerVariable(
        'float', 'outreach (relatability)', 'outreach US',
        'How relatable are the speaker’s arguments to the everyday experiences and concerns of a US citizen?',
    ),
    'outreach (accessibility)': SpeakerVariable(
        'float', 'outreach (accessibility)', 'outreach US',
        'Are the speaker’s arguments presented in an accessible and understandable manner to a wide audience in the USA?',
    ),
    'outreach (amplification)': SpeakerVariable(
        'float', 'outreach (amplification)', 'outreach US',
        'Are the speaker’s arguments likely to be amplified and spread by media and social platforms in the US?',
    ),
    'outreach (cultural relevance)': SpeakerVariable(
        'float', 'outreach (cultural relevance)', 'outreach US',
        'Do the speaker’s arguments align with the cultural values, norms, and contexts of the US?',
    ),
    'outreach (resonance)': SpeakerVariable(
        'float', 'outreach (resonance)', 'outreach US',
        'How well do the speaker’s arguments resonate with the emotions, values, and experiences of US citizens?',
    ),

    'logical (logic argument)': SpeakerVariable(
        'float', 'logical (logic argument)', 'logical',
        'How logical are the speakers arguments?',
    ),
    'logical (sound)': SpeakerVariable(
        'float', 'logical (sound)', 'logical',
        'Are the speakers arguments sound?',
    )
}

result_variables = {
    # general score
    'general score (argue)': SpeakerVariable(
        'float', 'general score (argue)', 'score',
        'How well does the speaker argue?',
    ),
    'general score (argument)': SpeakerVariable(
        'float', 'general score (argument)', 'score',
        'What is the quality of the speaker\'s arguments?',
    ),
    'general score (quality)': SpeakerVariable(
        'float', 'general score (quality)', 'score',
        'Do the speakers arguments improve the quality of the debate?',
    ),
    'general score (voting)': SpeakerVariable(
        'float', 'general score (voting)', 'score',
        'Do the speakers arguments increase the chance of winning the election?',
    ),

    # academic score
    'academic score (argue)': SpeakerVariable(
        'float', 'academic score (argue)', 'academic score',
        'Is the speakers argumentation structured well from an academic point of view?',
    ),
    'academic score (argument)': SpeakerVariable(
        'float', 'academic score (argument)', 'academic score',
        'What is the quality of the speaker\'s arguments from an academic point of view?',
    ),
    'academic score (academic)': SpeakerVariable(
        'float', 'academic score (structure)', 'academic score',
        'Does the speakers way of arguing follow the academic standards of argumentation?',
    ),

    # 'academic score (argue)': SpeakerObservable(
    #     'float', 'academic score (argue)', 'academic score',
    #     'How well does the speaker argue?',
    # ),
    # 'academic score (argument)': SpeakerObservable(
    #     'float', 'academic score (argument)', 'academic score',
    #     'What is the quality of the speaker\'s arguments?',
    # ),
    # 'academic score (academic)': SpeakerObservable(
    #     'float', 'academic score (academic)', 'academic score',
    #     'Do the speakers arguments improve the academic quality of the debate?',
    # ),

    # score for likelihood of winning the election?
    # 'election score (argue)': SpeakerObservable(
    #     'float', 'election score (argue)', 'election score',
    #     'How well does the speaker argue?',
    # ),
    # 'election score (argument)': SpeakerObservable(
    #     'float', 'election score (argument)', 'election score',
    #     'What is the quality of the speaker\'s arguments?',
    # ),
    'election score (voting)': SpeakerVariable(
        'float', 'election score (voting)', 'election score',
        'Do the speakers arguments increase the chance of winning the election?',
    ),
    'election score (election)': SpeakerVariable(
        'float', 'election score (election)', 'election score',
        'Based on the speaker\'s arguments, how likely is it that the speaker\'s party will win the election?',
    ),

    # election score for the US
    'US election score (argue)': SpeakerVariable(
        'float', 'US election score (argue)', 'US election score',
        'How well does the speaker argue?',
    ),
    'US election score (argument)': SpeakerVariable(
        'float', 'US election score (argument)', 'US election score',
        'What is the quality of the speaker\'s arguments?',
    ),
    'US election score (voting)': SpeakerVariable(
        'float', 'US election score (voting)', 'US election score',
        'Do the speakers arguments increase the chance of winning the election?',
    ),
    'US election score (election)': SpeakerVariable(
        'float', 'US election score (election)', 'US election score',
        'Based on the speaker\'s arguments, how likely is it that the speaker\'s party will win the election?',
    ),

    # score for likelihood of reaching the ears and minds of society?
    'society score (reach)': SpeakerVariable(
        'float', 'society score (reach)', 'society score',
        'Based on the speaker\'s arguments, how likely is it that the speaker\'s arguments will reach the ears and minds of society?',
    ),

    'pro democratic (argument)': SpeakerVariable(
        'float', 'pro democratic (argument)', 'pro democratic',
        'How democratic is the speaker\'s argument?',
    ),
    'pro republican (argument)': SpeakerVariable(
        'float', 'pro republican (argument)', 'pro republican',
        'How republican is the speaker\'s argument?',
    ),
    'pro neutral (argument)': SpeakerVariable(
        'float', 'pro neutral (argument)', 'pro neutral',
        'How neutral is the speaker\'s argument?',
    ),
    'pro democratic (benefit)': SpeakerVariable(
        'float', 'pro democratic (benefit)', 'pro democratic',
        'How much does the speaker benefit the democratic party?',
    ),
    'pro republican (benefit)': SpeakerVariable(
        'float', 'pro republican (benefit)', 'pro republican',
        'How much does the speaker benefit the republican party?',
    ),
    'pro neutral (benefit)': SpeakerVariable(
        'float', 'pro neutral (benefit)', 'pro neutral',
        'How much does the speaker benefit the neutral party?',
    ),

    'impact on audience (impact)': SpeakerVariable(
        'float', 'impact on audience (impact)', 'impact on audience',
        'How much potential does the speaker\'s arguments have to influence people\'s opinions or decisions?',
    ),

    'positive impact on audience (impact)': SpeakerVariable(
        'float', 'positive impact on audience (impact)', 'positive impact on audience',
        'How much potential does the speaker\'s arguments have to positively influence people\'s opinions or decisions?',
    ),

    'impact on economy (impact)': SpeakerVariable(
        'float', 'impact on economy (impact)', 'impact on economy',
        'How much does implementing the speaker\'s arguments affect the economy?',
    ),

    'positive impact on economy (impact)': SpeakerVariable(
        'float', 'positive impact on economy (impact)', 'positive impact on economy',
        'How much does implementing the speaker\'s arguments positively affect the economy?',
    ),

    'impact on society (impact)': SpeakerVariable(
        'float', 'impact on society (impact)', 'impact on society',
        'How much does implementing the speaker\'s arguments affect society?',
    ),

    'positive impact on society (impact)': SpeakerVariable(
        'float', 'positive impact on society (impact)', 'positive impact on society',
        'How much does implementing the speaker\'s arguments positively affect society?',
    ),

    'impact on environment (impact)': SpeakerVariable(
        'float', 'impact on environment (impact)', 'impact on environment',
        'How much does implementing the speaker\'s arguments affect the environment?',
    ),

    'positive impact on environment (impact)': SpeakerVariable(
        'float', 'positive impact on environment (impact)', 'positive impact on environment',
        'How much does implementing the speaker\'s arguments positively affect the environment?',
    ),

    'impact on politics (impact)': SpeakerVariable(
        'float', 'impact on politics (impact)', 'impact on politics',
        'How much does implementing the speaker\'s arguments affect politics?',
    ),

    'positive impact on politics (impact)': SpeakerVariable(
        'float', 'positive impact on politics (impact)', 'positive impact on politics',
        'How much does implementing the speaker\'s arguments positively affect politics?',
    ),

    'impact on rich population (impact)': SpeakerVariable(
        'float', 'impact on rich population (impact)', 'impact on rich population',
        'How much does implementing the speaker\'s arguments affect the rich population?',
    ),

    'positive impact on rich population (impact)': SpeakerVariable(
        'float', 'positive impact on rich population (impact)', 'positive impact on rich population',
        'How much does implementing the speaker\'s arguments positively affect the rich population?',
    ),

    'impact on poor population (impact)': SpeakerVariable(
        'float', 'impact on poor population (impact)', 'impact on poor population',
        'How much does implementing the speaker\'s arguments affect the poor population?',
    ),

    'positive impact on poor population (impact)': SpeakerVariable(
        'float', 'positive impact on poor population (impact)', 'positive impact on poor population',
        'How much does implementing the speaker\'s arguments positively affect the poor population?',
    ),

    'positive impact on USA (impact)': SpeakerVariable(
        'float', 'positive impact on USA (impact)', 'positive impact on USA',
        'How much does implementing the speaker\'s arguments positively affect the USA?',
    ),

    'positive impact on army funding (impact)': SpeakerVariable(
        'float', 'positive impact on army funding (impact)', 'positive impact on army funding',
        'How much does implementing the speaker\'s arguments positively affect army funding?',
    ),

    'positive impact on China (impact)': SpeakerVariable(
        'float', 'positive impact on China (impact)', 'positive impact on China',
        'How much does implementing the speaker\'s arguments positively affect China?',
    ),

    'positive impact on Russia (impact)': SpeakerVariable(
        'float', 'positive impact on Russia (impact)', 'positive impact on Russia',
        'How much does implementing the speaker\'s arguments positively affect Russia?',
    ),

    'positive impact on Western Europe (impact)': SpeakerVariable(
        'float', 'positive impact on Western Europe (impact)', 'positive impact on Western Europe',
        'How much does implementing the speaker\'s arguments positively affect Western Europe?',
    ),

    'positive impact on World (impact)': SpeakerVariable(
        'float', 'positive impact on World (impact)', 'positive impact on World',
        'How much does implementing the speaker\'s arguments positively affect the World?',
    ),

    'positive impact on Middle East (impact)': SpeakerVariable(
        'float', 'positive impact on Middle East (impact)', 'positive impact on Middle East',
        'How much does implementing the speaker\'s arguments positively affect the Middle East?',
    ),
}

speaker_predictor_variables_groups = {}
for variable in predictor_variables.values():
    speaker_predictor_variables_groups[variable.name] = speaker_predictor_variables_groups.get(variable.name, []) + [variable]

# there was an error in the naming of relevant and relevance, combine them now...
speaker_predictor_variables_groups['relevance'] += [v for v in speaker_predictor_variables_groups['relevant']]
for v in speaker_predictor_variables_groups['relevance']:
    v.name = 'relevance'
del speaker_predictor_variables_groups['relevant']

speaker_result_variables_groups = {}
for variable in result_variables.values():
    speaker_result_variables_groups[variable.name] = speaker_result_variables_groups.get(variable.name, []) + [variable]


speaker_variables = {
    **predictor_variables,
    **result_variables,
}

speaker_variables_groups = {}
for variable in speaker_variables.values():
    speaker_variables_groups[variable.name] = speaker_variables_groups.get(variable.name, []) + [variable]


multi_speaker_variables_groups = {
    'score contribution group 1': [
        speaker_variables['contribution (quality)'],
        speaker_variables['general score (argue)'],
    ],
    'score contribution group 1 inverse': [
        speaker_variables['general score (argue)'],
        speaker_variables['contribution (quality)'],
    ],
    'score contribution group 2': [
        speaker_variables['contribution (quality)'],
        speaker_variables['general score (argument)'],
    ],
    'score contribution group 2 inverse': [
        speaker_variables['general score (argument)'],
        speaker_variables['contribution (quality)'],
    ],
    'group 1': [
        speaker_variables['pro democratic (argument)'],
        speaker_variables['egotistical (benefit)'],
        speaker_variables['persuasiveness (convincing)'],
        speaker_variables['clarity (understandable)'],
    ],
    'group 1 inverse': [
        speaker_variables['clarity (understandable)'],
        speaker_variables['persuasiveness (convincing)'],
        speaker_variables['egotistical (benefit)'],
        speaker_variables['pro democratic (argument)'],
    ],
    'group 2': [
        speaker_variables['pro republican (argument)'],
        speaker_variables['persuasiveness (convincing)'],
        speaker_variables['clarity (easiness)'],
        speaker_variables['contribution (quality)'],
    ],
    'group 2 inverse': [
        speaker_variables['contribution (quality)'],
        speaker_variables['clarity (easiness)'],
        speaker_variables['persuasiveness (convincing)'],
        speaker_variables['pro republican (argument)'],
    ],
}

slice_speaker_variables_correlated_by_design = [
    [
        'pro democratic',
        'pro republican',
        'speaker_party_is_DEMOCRAT',
        'speaker_party_is_REPUBLICAN',
    ],
    [
        'speaker_quantitative_contribution',
        'speaker_quantitative_contribution_ratio',
    ],
    [
        'speaker_num_parts',
        'speaker_avg_part_size',
    ],
    [
        'slice_size',
        'speaker_avg_part_size',
    ],
]
