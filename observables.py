from prompt_builder import SliceObservable, SingleSliceObservablePrompt, MultiSliceObservablePrompt, SpeakerObservable, MultiSpeakerObservablePrompt, MultiSpeakerObservableMultiSpeakersPrompt


defined_observables_descriptions = {
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

slice_observables = {
    'content quality (filler)': SliceObservable(
        'float', 'content quality (filler)', 'content quality',
        'Is there any content in this part of the debate or is it mostly filler?',
    ),
    'content quality (speaker)': SliceObservable(
        'float', 'content quality (speaker)', 'content quality',
        'Is there any valuable content in this part of the debate that can be used for further analysis of how well the speakers can argue their points?',
    ),
    'content quality (dataset)': SliceObservable(
        'float', 'content quality (dataset)', 'content quality',
        'We want to create a dataset to study how well the speakers can argue, convery information and what leads to winning an election. Should this part of the debate be included in the dataset?',
    ),
    'topic predictiveness (usefullness)': SliceObservable(
        'float', 'topic predictiveness (usefullness)', 'topic predictiveness',
        'Can this part of the debate be used to predict the topic of the debate?',
    ),
    'topic (max3)': SliceObservable(
        'str', 'topic (max3)', 'topic',
        'Which topic is being discussed in this part of the debate? Respond with a short, compact and general title with max 3 words in all caps.'
    ),
}

slice_observables_groups = {}
for observable in slice_observables.values():
    slice_observables_groups[observable.name] = slice_observables_groups.get(observable.name, []) + [observable]

predictor_observables = {
    'egotistical (benefit)': SpeakerObservable(
        'float', 'egotistical (benefit)', 'egotistical',
        'How much do the speaker\'s arguments benefit the speaker himself?',
    ),

    'persuasiveness (convincing)': SpeakerObservable(
        'float', 'persuasiveness (convincing)', 'persuasiveness',
        'How convincing are the arguments or points made by the speaker?',
    ),

    'clarity (understandable)': SpeakerObservable(
        'float', 'clarity (understandable)', 'clarity',
        'How clear and understandable is the speaker\'s arguments?',
    ),
    'clarity (easiness)': SpeakerObservable(
        'float', 'clarity (easiness)', 'clarity',
        'How easy are the speaker\'s arguments to understand for a general audience?',
    ),
    'clarity (clarity)': SpeakerObservable(
        'float', 'clarity (clarity)', 'clarity',
        'Is the speaker able to convey their arguments in a clear and comprehensible manner?',
    ),


    'contribution (quality)': SpeakerObservable(
        'float', 'contribution (quality)', 'contribution',
        'How good is the speaker\'s contribution to the discussion?',
    ),
    'contribution (quantity)': SpeakerObservable(
        'float', 'contribution (quantity)', 'contribution',
        'How much does the speaker contribute to the discussion?',
    ),

    'truthfulness (thruthullness)': SpeakerObservable(
        'float', 'truthfulness (thruthullness)', 'truthfulness',
        'How truthful are the speaker\'s arguments?',
    ),
    # 'truthfulness (lie)': SpeakerObservable(
    #     'float', 'truthfulness (lie)', 'truthfulness',
    #     'How much does the speaker lie?',
    # ), # always gave 0.5 and was not what we wanted...

    'bias (bias)': SpeakerObservable(
        'float', 'bias (bias)', 'bias',
        'How biased is the speaker?',
    ),

    'manipulation (manipulation)': SpeakerObservable(
        'float', 'manipulation (manipulation)', 'manipulation',
        'Is the speaker trying to subtly guide the reader towards a particular conclusion or opinion?',
    ),
    'manipulation (underhanded)': SpeakerObservable(
        'float', 'manipulation (underhanded)', 'manipulation',
        'Is the speaker trying to underhandedly guide the reader towards a particular conclusion or opinion?',
    ),

    'evasiveness (avoid)': SpeakerObservable(
        'float', 'evasiveness (avoid)', 'evasiveness',
        'Does the speaker avoid answering questions or addressing certain topics?',
    ),
    'evasiveness (ignore)': SpeakerObservable(
        'float', 'evasiveness (ignore)', 'evasiveness',
        'Does the speaker ignore certain topics or questions?',
    ),
    'evasiveness (dodge)': SpeakerObservable(
        'float', 'evasiveness (dodge)', 'evasiveness',
        'Does the speaker dodge certain topics or questions?',
    ),
    'evasiveness (evade)': SpeakerObservable(
        'float', 'evasiveness (evade)', 'evasiveness',
        'Does the speaker evade certain topics or questions?',
    ),

    'relevant (relevant)': SpeakerObservable(
        'float', 'relevant (relevant)', 'relevant',
        'How relevant is the speaker\'s arguments to the stated topic or subject?',
    ),

    'everyday relevance (relevance)': SpeakerObservable(
        'float', 'relevance (relevance)', 'relevance',
        'Do the speaker’s arguments and issues addressed have relevance to the everyday lives of the audience?',
    ),

    'conciseness (efficiency)': SpeakerObservable(
        'float', 'conciseness (efficiency)', 'conciseness',
        'Does the speaker express his points efficiently without unnecessary verbiage?',
    ),
    'conciseness (concise)': SpeakerObservable(
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

    'use of evidence (evidence)': SpeakerObservable(
        'float', 'use of evidence (evidence)', 'use of evidence',
        'Does the speaker use solid evidence to support his points?',
    ),

    'emotional appeal (emotional)': SpeakerObservable(
        'float', 'emotional appeal (emotional)', 'emotional appeal',
        'Does the speaker use emotional language or appeals to sway the reader?',
    ),

    'objectivity (unbiased)': SpeakerObservable(
        'float', 'objectivity (unbiased)', 'objectivity',
        'Does the speaker attempt to present an unbiased, objective view of the topic?',
    ),

    'sensationalism (exaggerated)': SpeakerObservable(
        'float', 'sensationalism (exaggerated)', 'sensationalism',
        'Does the speaker use exaggerated or sensational language to attract attention?',
    ),

    'controversiality (controversial)': SpeakerObservable(
        'float', 'controversiality (controversial)', 'controversiality',
        'Does the speaker touch on controversial topics or take controversial stances?',
    ),

    'coherence (coherent)': SpeakerObservable(
        'float', 'coherence (coherent)', 'coherence',
        'Do the speaker\'s points logically follow from one another?',
    ),

    'consistency (consistent)': SpeakerObservable(
        'float', 'consistency (consistent)', 'consistency',
        'Are the arguments and viewpoints the speaker presents consistent with each other?',
    ),

    # 'originality (original)': SpeakerObservable(
    #     'float', 'originality (original)', 'originality',
    #     'Does the speaker present new ideas or perspectives, or does he rehash existing ones?',
    # ), # always gave 0.5

    'factuality (factual)': SpeakerObservable(
        'float', 'factuality (factual)', 'factuality',
        'How much of the speaker\'s arguments are based on factual information versus opinion?',
    ),

    'completeness (complete)': SpeakerObservable(
        'float', 'completeness (complete)', 'completeness',
        'Does the speaker cover the topic fully and address all relevant aspects?',
    ),

    'quality of sources (reliable)': SpeakerObservable(
        'float', 'quality of sources (reliable)', 'quality of sources',
        'How reliable and credible are the sources used by the speaker?',
    ),

    'balance (balanced)': SpeakerObservable(
        'float', 'balance (balanced)', 'balance',
        'Does the speaker present multiple sides of the issue, or is it one-sided?',
    ),

    'tone is professional (tone)': SpeakerObservable(
        'float', 'tone is professional (tone)', 'tone is professional',
        'Does the speaker use a professional tone?',
    ),
    'tone is conversational (tone)': SpeakerObservable(
        'float', 'tone is conversational (tone)', 'tone is conversational',
        'Does the speaker use a conversational tone?',
    ),
    'tone is academic (tone)': SpeakerObservable(
        'float', 'tone is academic (tone)', 'tone is academic',
        'Does the speaker use an academic tone?',
    ),

    'accessibility (accessibility)': SpeakerObservable(
        'float', 'accessibility (accessibility)', 'accessibility',
        'How easily can the speaker be understood by a general audience?',
    ),

    'engagement (engagement)': SpeakerObservable(
        'float', 'engagement (engagement)', 'engagement',
        'How much does the speaker draw in and hold the reader\'s attention?',
    ),
    'engagement (participation)': SpeakerObservable(
        'float', 'engagement (engagement)', 'engagement',
        'Does the speaker actively engage the audience, encouraging participation and dialogue?',
    ),

    'adherence to rules (adherence)': SpeakerObservable(
        'float', 'adherence to rules (adherence)', 'adherence to rules',
        'Does the speaker respect and adhere to the rules and format of the debate or discussion?',
    ),

    'respectfulness (respectfulness)': SpeakerObservable(
        'float', 'respectfulness (respectfulness)', 'respectfulness',
        'Does the speaker show respect to others involved in the discussion, including the moderator and other participants?',
    ),

    'interruptions (interruptions)': SpeakerObservable(
        'float', 'interruptions (interruptions)', 'interruptions',
        'How often does the speaker interrupt others when they are speaking?',
    ),

    'time management (time management)': SpeakerObservable(
        'float', 'time management (time management)', 'time management',
        'Does the speaker make effective use of their allotted time, and respect the time limits set for their responses?',
    ),

    'responsiveness (responsiveness)': SpeakerObservable(
        'float', 'responsiveness (responsiveness)', 'responsiveness',
        'How directly does the speaker respond to questions or prompts from the moderator or other participants?',
    ),

    'decorum (decorum)': SpeakerObservable(
        'float', 'decorum (decorum)', 'decorum',
        'Does the speaker maintain the level of decorum expected in the context of the discussion?',
    ),

    'venue respect (venue respect)': SpeakerObservable(
        'float', 'venue respect (venue respect)', 'venue respect',
        'Does the speaker show respect for the venue and event where the debate is held?',
    ),

    'language appropriateness (language appropriateness)': SpeakerObservable(
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

    'contextual awareness (contextual awareness)': SpeakerObservable(
        'float', 'contextual awareness (contextual awareness)', 'contextual awareness',
        'How much does the speaker demonstrate awareness of the context of the discussion?',
    ),

    'confidence (confidence)': SpeakerObservable(
        'float', 'confidence (confidence)', 'confidence',
        'How confident does the speaker appear?',
    ),

    'fair play (fair play)': SpeakerObservable(
        'float', 'fair play (fair play)', 'fair play',
        'Does the speaker engage in fair debating tactics, or do they resort to logical fallacies, personal attacks, or other unfair tactics?',
    ),

    'listening skills (listening skills)': SpeakerObservable(
        'float', 'listening skills (listening skills)', 'listening skills',
        'Does the speaker show that they are actively listening and responding to the points made by others?',
    ),

    'civil discourse (civil discourse)': SpeakerObservable(
        'float', 'civil discourse (civil discourse)', 'civil discourse',
        'Does the speaker contribute to maintaining a climate of civil discourse, where all participants feel respected and heard?',
    ),

    'respect for diverse opinions (respect for diverse opinions)': SpeakerObservable(
        'float', 'respect for diverse opinions (respect for diverse opinions)', 'respect for diverse opinions',
        'Does the speaker show respect for viewpoints different from their own, even while arguing against them?',
    ),

    'preparation (preparation)': SpeakerObservable(
        'float', 'preparation (preparation)', 'preparation',
        'Does the speaker seem well-prepared for the debate, demonstrating a good understanding of the topics and questions at hand?',
    ),

    'resonance (resonance)': SpeakerObservable(
        'float', 'resonance (resonance)', 'resonance',
        'Does the speaker’s message resonate with the audience, aligning with their values, experiences, and emotions?',
    ),

    'authenticity (authenticity)': SpeakerObservable(
        'float', 'authenticity (authenticity)', 'authenticity',
        'Does the speaker come across as genuine and authentic in their communication and representation of issues?',
    ),

    'empathy (empathy)': SpeakerObservable(
        'float', 'empathy (empathy)', 'empathy',
        'Does the speaker demonstrate empathy and understanding towards the concerns and needs of the audience?',
    ),

    'innovation (innovation)': SpeakerObservable(
        'float', 'innovation (innovation)', 'innovation',
        'Does the speaker introduce innovative ideas and perspectives that contribute to the discourse?',
    ),

    'outreach (penetration)': SpeakerObservable(
        'float', 'outreach (penetration)', 'outreach US',
        'How effectively do the speaker’s arguments penetrate various demographics and social groups within the US society?',
    ),
    'outreach (relatability)': SpeakerObservable(
        'float', 'outreach (relatability)', 'outreach US',
        'How relatable are the speaker’s arguments to the everyday experiences and concerns of a US citizen?',
    ),
    'outreach (accessibility)': SpeakerObservable(
        'float', 'outreach (accessibility)', 'outreach US',
        'Are the speaker’s arguments presented in an accessible and understandable manner to a wide audience in the USA?',
    ),
    'outreach (amplification)': SpeakerObservable(
        'float', 'outreach (amplification)', 'outreach US',
        'Are the speaker’s arguments likely to be amplified and spread by media and social platforms in the US?',
    ),
    'outreach (cultural relevance)': SpeakerObservable(
        'float', 'outreach (cultural relevance)', 'outreach US',
        'Do the speaker’s arguments align with the cultural values, norms, and contexts of the US?',
    ),
    'outreach (resonance)': SpeakerObservable(
        'float', 'outreach (resonance)', 'outreach US',
        'How well do the speaker’s arguments resonate with the emotions, values, and experiences of US citizens?',
    ),

    'logical (logic argument)': SpeakerObservable(
        'float', 'logical (logic argument)', 'logical',
        'How logical are the speakers arguments?',
    ),
    'logical (sound)': SpeakerObservable(
        'float', 'logical (sound)', 'logical',
        'Are the speakers arguments sound?',
    )
}

result_observables = {
    # general score
    'general score (argue)': SpeakerObservable(
        'float', 'general score (argue)', 'score',
        'How well does the speaker argue?',
    ),
    'general score (argument)': SpeakerObservable(
        'float', 'general score (argument)', 'score',
        'What is the quality of the speaker\'s arguments?',
    ),
    'general score (quality)': SpeakerObservable(
        'float', 'general score (quality)', 'score',
        'Do the speakers arguments improve the quality of the debate?',
    ),
    'general score (voting)': SpeakerObservable(
        'float', 'general score (voting)', 'score',
        'Do the speakers arguments increase the chance of winning the election?',
    ),

    # academic score
    'academic score (argue)': SpeakerObservable(
        'float', 'academic score (argue)', 'academic score',
        'Is the speakers argumentation structured well from an academic point of view?',
    ),
    'academic score (argument)': SpeakerObservable(
        'float', 'academic score (argument)', 'academic score',
        'What is the quality of the speaker\'s arguments from an academic point of view?',
    ),
    'academic score (academic)': SpeakerObservable(
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
    'election score (voting)': SpeakerObservable(
        'float', 'election score (voting)', 'election score',
        'Do the speakers arguments increase the chance of winning the election?',
    ),
    'election score (election)': SpeakerObservable(
        'float', 'election score (election)', 'election score',
        'Based on the speaker\'s arguments, how likely is it that the speaker\'s party will win the election?',
    ),

    # election score for the US
    'US election score (argue)': SpeakerObservable(
        'float', 'US election score (argue)', 'US election score',
        'How well does the speaker argue?',
    ),
    'US election score (argument)': SpeakerObservable(
        'float', 'US election score (argument)', 'US election score',
        'What is the quality of the speaker\'s arguments?',
    ),
    'US election score (voting)': SpeakerObservable(
        'float', 'US election score (voting)', 'US election score',
        'Do the speakers arguments increase the chance of winning the election?',
    ),
    'US election score (election)': SpeakerObservable(
        'float', 'US election score (election)', 'US election score',
        'Based on the speaker\'s arguments, how likely is it that the speaker\'s party will win the election?',
    ),

    # score for likelihood of reaching the ears and minds of society?
    'society score (reach)': SpeakerObservable(
        'float', 'society score (reach)', 'society score',
        'Based on the speaker\'s arguments, how likely is it that the speaker\'s arguments will reach the ears and minds of society?',
    ),

    'pro democratic (argument)': SpeakerObservable(
        'float', 'pro democratic (argument)', 'pro democratic',
        'How democratic is the speaker\'s argument?',
    ),
    'pro republican (argument)': SpeakerObservable(
        'float', 'pro republican (argument)', 'pro republican',
        'How republican is the speaker\'s argument?',
    ),
    'pro neutral (argument)': SpeakerObservable(
        'float', 'pro neutral (argument)', 'pro neutral',
        'How neutral is the speaker\'s argument?',
    ),
    'pro democratic (benefit)': SpeakerObservable(
        'float', 'pro democratic (benefit)', 'pro democratic',
        'How much does the speaker benefit the democratic party?',
    ),
    'pro republican (benefit)': SpeakerObservable(
        'float', 'pro republican (benefit)', 'pro republican',
        'How much does the speaker benefit the republican party?',
    ),
    'pro neutral (benefit)': SpeakerObservable(
        'float', 'pro neutral (benefit)', 'pro neutral',
        'How much does the speaker benefit the neutral party?',
    ),

    'impact on audience (impact)': SpeakerObservable(
        'float', 'impact on audience (impact)', 'impact on audience',
        'How much potential does the speaker\'s arguments have to influence people\'s opinions or decisions?',
    ),

    'positive impact on audience (impact)': SpeakerObservable(
        'float', 'positive impact on audience (impact)', 'positive impact on audience',
        'How much potential does the speaker\'s arguments have to positively influence people\'s opinions or decisions?',
    ),

    'impact on economy (impact)': SpeakerObservable(
        'float', 'impact on economy (impact)', 'impact on economy',
        'How much does implementing the speaker\'s arguments affect the economy?',
    ),

    'positive impact on economy (impact)': SpeakerObservable(
        'float', 'positive impact on economy (impact)', 'positive impact on economy',
        'How much does implementing the speaker\'s arguments positively affect the economy?',
    ),

    'impact on society (impact)': SpeakerObservable(
        'float', 'impact on society (impact)', 'impact on society',
        'How much does implementing the speaker\'s arguments affect society?',
    ),

    'positive impact on society (impact)': SpeakerObservable(
        'float', 'positive impact on society (impact)', 'positive impact on society',
        'How much does implementing the speaker\'s arguments positively affect society?',
    ),

    'impact on environment (impact)': SpeakerObservable(
        'float', 'impact on environment (impact)', 'impact on environment',
        'How much does implementing the speaker\'s arguments affect the environment?',
    ),

    'positive impact on environment (impact)': SpeakerObservable(
        'float', 'positive impact on environment (impact)', 'positive impact on environment',
        'How much does implementing the speaker\'s arguments positively affect the environment?',
    ),

    'impact on politics (impact)': SpeakerObservable(
        'float', 'impact on politics (impact)', 'impact on politics',
        'How much does implementing the speaker\'s arguments affect politics?',
    ),

    'positive impact on politics (impact)': SpeakerObservable(
        'float', 'positive impact on politics (impact)', 'positive impact on politics',
        'How much does implementing the speaker\'s arguments positively affect politics?',
    ),

    'impact on rich population (impact)': SpeakerObservable(
        'float', 'impact on rich population (impact)', 'impact on rich population',
        'How much does implementing the speaker\'s arguments affect the rich population?',
    ),

    'positive impact on rich population (impact)': SpeakerObservable(
        'float', 'positive impact on rich population (impact)', 'positive impact on rich population',
        'How much does implementing the speaker\'s arguments positively affect the rich population?',
    ),

    'impact on poor population (impact)': SpeakerObservable(
        'float', 'impact on poor population (impact)', 'impact on poor population',
        'How much does implementing the speaker\'s arguments affect the poor population?',
    ),

    'positive impact on poor population (impact)': SpeakerObservable(
        'float', 'positive impact on poor population (impact)', 'positive impact on poor population',
        'How much does implementing the speaker\'s arguments positively affect the poor population?',
    ),

    'positive impact on USA (impact)': SpeakerObservable(
        'float', 'positive impact on USA (impact)', 'positive impact on USA',
        'How much does implementing the speaker\'s arguments positively affect the USA?',
    ),

    'positive impact on army funding (impact)': SpeakerObservable(
        'float', 'positive impact on army funding (impact)', 'positive impact on army funding',
        'How much does implementing the speaker\'s arguments positively affect army funding?',
    ),

    'positive impact on China (impact)': SpeakerObservable(
        'float', 'positive impact on China (impact)', 'positive impact on China',
        'How much does implementing the speaker\'s arguments positively affect China?',
    ),

    'positive impact on Russia (impact)': SpeakerObservable(
        'float', 'positive impact on Russia (impact)', 'positive impact on Russia',
        'How much does implementing the speaker\'s arguments positively affect Russia?',
    ),

    'positive impact on Western Europe (impact)': SpeakerObservable(
        'float', 'positive impact on Western Europe (impact)', 'positive impact on Western Europe',
        'How much does implementing the speaker\'s arguments positively affect Western Europe?',
    ),

    'positive impact on World (impact)': SpeakerObservable(
        'float', 'positive impact on World (impact)', 'positive impact on World',
        'How much does implementing the speaker\'s arguments positively affect the World?',
    ),

    'positive impact on Middle East (impact)': SpeakerObservable(
        'float', 'positive impact on Middle East (impact)', 'positive impact on Middle East',
        'How much does implementing the speaker\'s arguments positively affect the Middle East?',
    ),
}

speaker_predictor_observables_groups = {}
for observable in predictor_observables.values():
    speaker_predictor_observables_groups[observable.name] = speaker_predictor_observables_groups.get(observable.name, []) + [observable]

# there was an error in the naming of relevant and relevance, combine them now...
speaker_predictor_observables_groups['relevance'] += [v for v in speaker_predictor_observables_groups['relevant']]
for v in speaker_predictor_observables_groups['relevance']:
    v.name = 'relevance'
del speaker_predictor_observables_groups['relevant']

speaker_result_observables_groups = {}
for observable in result_observables.values():
    speaker_result_observables_groups[observable.name] = speaker_result_observables_groups.get(observable.name, []) + [observable]


speaker_observables = {
    **predictor_observables,
    **result_observables,
}

speaker_observables_groups = {}
for observable in speaker_observables.values():
    speaker_observables_groups[observable.name] = speaker_observables_groups.get(observable.name, []) + [observable]


multi_speaker_observables_groups = {
    'score contribution group 1': [
        speaker_observables['contribution (quality)'],
        speaker_observables['general score (argue)'],
    ],
    'score contribution group 1 inverse': [
        speaker_observables['general score (argue)'],
        speaker_observables['contribution (quality)'],
    ],
    'score contribution group 2': [
        speaker_observables['contribution (quality)'],
        speaker_observables['general score (argument)'],
    ],
    'score contribution group 2 inverse': [
        speaker_observables['general score (argument)'],
        speaker_observables['contribution (quality)'],
    ],
    'group 1': [
        speaker_observables['pro democratic (argument)'],
        speaker_observables['egotistical (benefit)'],
        speaker_observables['persuasiveness (convincing)'],
        speaker_observables['clarity (understandable)'],
    ],
    'group 1 inverse': [
        speaker_observables['clarity (understandable)'],
        speaker_observables['persuasiveness (convincing)'],
        speaker_observables['egotistical (benefit)'],
        speaker_observables['pro democratic (argument)'],
    ],
    'group 2': [
        speaker_observables['pro republican (argument)'],
        speaker_observables['persuasiveness (convincing)'],
        speaker_observables['clarity (easiness)'],
        speaker_observables['contribution (quality)'],
    ],
    'group 2 inverse': [
        speaker_observables['contribution (quality)'],
        speaker_observables['clarity (easiness)'],
        speaker_observables['persuasiveness (convincing)'],
        speaker_observables['pro republican (argument)'],
    ],
}

slice_speaker_observables_correlated_by_design = [
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
