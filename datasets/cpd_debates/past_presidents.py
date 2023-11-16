# from https://www.infoplease.com/us/government/elections/presidential-elections-1789-2020
# converted using gpt 4 in july 2023
# checked by hand

year_to_presidents = {
    '1960': {'elected_president': 'JOHN KENNEDY',
                               'elected_vice_president': 'LYNDON JOHNSON',
                               'electoral_votes': {'DEMOCRAT': 303, 'REPUBLICAN': 219},
                               'popular_votes': {'DEMOCRAT': 34200000, 'REPUBLICAN': 34100000},
                               'president_candidates': {'DEMOCRAT': 'JOHN KENNEDY',
                                                        'REPUBLICAN': 'RICHARD NIXON'},
                               'vice_president_candidates': {'DEMOCRAT': 'LYNDON JOHNSON',
                                                             'REPUBLICAN': 'HENRY CABOT LODGE'}},
                      '1964': {'elected_president': 'LYNDON JOHNSON',
                               'elected_vice_president': 'HUBERT HUMPHREY',
                               'electoral_votes': {'DEMOCRAT': 486, 'REPUBLICAN': 52},
                               'popular_votes': {'DEMOCRAT': 43100000, 'REPUBLICAN': 27200000},
                               'president_candidates': {'DEMOCRAT': 'LYNDON JOHNSON',
                                                        'REPUBLICAN': 'BARRY GOLDWATER'},
                               'vice_president_candidates': {'DEMOCRAT': 'HUBERT HUMPHREY',
                                                             'REPUBLICAN': 'WILLIAM MILLER'}},
                      '1968': {'elected_president': 'RICHARD NIXON',
                               'elected_vice_president': 'SPIRO AGNEW',
                               'electoral_votes': {'DEMOCRAT': 191, 'REPUBLICAN': 301},
                               'popular_votes': {'DEMOCRAT': 31300000, 'REPUBLICAN': 31800000},
                               'president_candidates': {'DEMOCRAT': 'HUBERT HUMPHREY',
                                                        'REPUBLICAN': 'RICHARD NIXON'},
                               'vice_president_candidates': {'DEMOCRAT': 'EDMUND MUSKIE',
                                                             'REPUBLICAN': 'SPIRO AGNEW'}},
                      '1972': {'elected_president': 'RICHARD NIXON',
                               'elected_vice_president': 'SPIRO AGNEW',
                               'electoral_votes': {'DEMOCRAT': 17, 'REPUBLICAN': 520},
                               'popular_votes': {'DEMOCRAT': 29200000, 'REPUBLICAN': 47200000},
                               'president_candidates': {'DEMOCRAT': 'GEORGE MCGOVERN',
                                                        'REPUBLICAN': 'RICHARD NIXON'},
                               'vice_president_candidates': {'DEMOCRAT': 'SARGENT SHRIVER',
                                                             'REPUBLICAN': 'SPIRO AGNEW'}},
                      '1976': {'elected_president': 'JIMMY CARTER',
                               'elected_vice_president': 'WALTER MONDALE',
                               'electoral_votes': {'DEMOCRAT': 297, 'REPUBLICAN': 240},
                               'popular_votes': {'DEMOCRAT': 40800000, 'REPUBLICAN': 39100000},
                               'president_candidates': {'DEMOCRAT': 'JIMMY CARTER',
                                                        'REPUBLICAN': 'GERALD FORD'},
                               'vice_president_candidates': {'DEMOCRAT': 'WALTER MONDALE',
                                                             'REPUBLICAN': 'ROBERT DOLE'}},
                      '1980': {'elected_president': 'RONALD REAGAN',
                               'elected_vice_president': 'GEORGE H. BUSH',
                               'electoral_votes': {'DEMOCRAT': 49, 'REPUBLICAN': 489},
                               'popular_votes': {'DEMOCRAT': 36500000, 'REPUBLICAN': 43900000},
                               'president_candidates': {'DEMOCRAT': 'JIMMY CARTER',
                                                        'REPUBLICAN': 'RONALD REAGAN'},
                               'vice_president_candidates': {'DEMOCRAT': 'WALTER MONDALE',
                                                             'REPUBLICAN': 'GEORGE H. BUSH'}},
                      '1984': {'elected_president': 'RONALD REAGAN',
                               'elected_vice_president': 'GEORGE H. BUSH',
                               'electoral_votes': {'DEMOCRAT': 13, 'REPUBLICAN': 525},
                               'popular_votes': {'DEMOCRAT': 37600000, 'REPUBLICAN': 54500000},
                               'president_candidates': {'DEMOCRAT': 'WALTER MONDALE',
                                                        'REPUBLICAN': 'RONALD REAGAN'},
                               'vice_president_candidates': {'DEMOCRAT': 'GERALDINE FERRARO',
                                                             'REPUBLICAN': 'GEORGE H. BUSH'}},
                      '1988': {'elected_president': 'GEORGE H. BUSH',
                               'elected_vice_president': 'DANFORTH QUAYLE',
                               'electoral_votes': {'DEMOCRAT': 111, 'REPUBLICAN': 426},
                               'popular_votes': {'DEMOCRAT': 41800000, 'REPUBLICAN': 48900000},
                               'president_candidates': {'DEMOCRAT': 'MICHAEL DUKAKIS',
                                                        'REPUBLICAN': 'GEORGE H. BUSH'},
                               'vice_president_candidates': {'DEMOCRAT': 'LLOYD BENTSEN',
                                                             'REPUBLICAN': 'DANFORTH QUAYLE'}},
                      '1992': {'elected_president': 'WILLIAM CLINTON',
                               'elected_vice_president': 'ALBERT GORE',
                               'electoral_votes': {'DEMOCRAT': 370, 'REPUBLICAN': 168},
                               'popular_votes': {'DEMOCRAT': 44900000, 'REPUBLICAN': 39100000},
                               'president_candidates': {'DEMOCRAT': 'WILLIAM CLINTON',
                                                        'REPUBLICAN': 'GEORGE H. BUSH'},
                               'vice_president_candidates': {'DEMOCRAT': 'ALBERT GORE',
                                                             'REPUBLICAN': 'DANFORTH QUAYLE'}},
                      '1996': {'elected_president': 'WILLIAM CLINTON',
                               'elected_vice_president': 'ALBERT GORE.',
                               'electoral_votes': {'DEMOCRAT': 379, 'REPUBLICAN': 159},
                               'popular_votes': {'DEMOCRAT': 47400000, 'REPUBLICAN': 39200000},
                               'president_candidates': {'DEMOCRAT': 'WILLIAM CLINTON',
                                                        'REPUBLICAN': 'ROBERT DOLE'},
                               'vice_president_candidates': {'DEMOCRAT': 'ALBERT GORE.',
                                                             'REPUBLICAN': 'JACK KEMP'}},
                      '2000': {'elected_president': 'GEORGE W. BUSH',
                               'elected_vice_president': 'RICHARD CHENEY',
                               'electoral_votes': {'DEMOCRAT': 266, 'REPUBLICAN': 271},
                               'popular_votes': {'DEMOCRAT': 51000000, 'REPUBLICAN': 50500000},
                               'president_candidates': {'DEMOCRAT': 'ALBERT GORE',
                                                        'REPUBLICAN': 'GEORGE W. BUSH'},
                               'vice_president_candidates': {'DEMOCRAT': 'JOSEPH LIEBERMAN',
                                                             'REPUBLICAN': 'RICHARD CHENEY'}},
                      '2004': {'elected_president': 'GEORGE W. BUSH',
                               'elected_vice_president': 'RICHARD CHENEY',
                               'electoral_votes': {'DEMOCRAT': 251, 'REPUBLICAN': 286},
                               'popular_votes': {'DEMOCRAT': 59000000, 'REPUBLICAN': 62000000},
                               'president_candidates': {'DEMOCRAT': 'JOHN KERRY',
                                                        'REPUBLICAN': 'GEORGE W. BUSH'},
                               'vice_president_candidates': {'DEMOCRAT': 'JOHN EDWARDS',
                                                             'REPUBLICAN': 'RICHARD CHENEY'}},
                      '2008': {'elected_president': 'BARACK OBAMA',
                               'elected_vice_president': 'JOE BIDEN',
                               'electoral_votes': {'DEMOCRAT': 365, 'REPUBLICAN': 173},
                               'popular_votes': {'DEMOCRAT': 66900000, 'REPUBLICAN': 58300000},
                               'president_candidates': {'DEMOCRAT': 'BARACK OBAMA',
                                                        'REPUBLICAN': 'JOHN MCCAIN'},
                               'vice_president_candidates': {'DEMOCRAT': 'JOE BIDEN',
                                                             'REPUBLICAN': 'SARAH PALIN'}},
                      '2012': {'elected_president': 'BARACK OBAMA',
                               'elected_vice_president': 'JOE BIDEN',
                               'electoral_votes': {'DEMOCRAT': 332, 'REPUBLICAN': 206},
                               'popular_votes': {'DEMOCRAT': 62600000, 'REPUBLICAN': 59100000},
                               'president_candidates': {'DEMOCRAT': 'BARACK OBAMA',
                                                        'REPUBLICAN': 'MITT ROMNEY'},
                               'vice_president_candidates': {'DEMOCRAT': 'JOE BIDEN',
                                                             'REPUBLICAN': 'PAUL RYAN'}},
                      '2016': {'elected_president': 'DONALD TRUMP',
                               'elected_vice_president': 'MICHAEL PENCE',
                               'electoral_votes': {'DEMOCRAT': 232, 'REPUBLICAN': 306},
                               'popular_votes': {'DEMOCRAT': 65800000, 'REPUBLICAN': 63000000},
                               'president_candidates': {'DEMOCRAT': 'HILLARY CLINTON',
                                                        'REPUBLICAN': 'DONALD TRUMP'},
                               'vice_president_candidates': {'DEMOCRAT': 'TIM KAINE',
                                                             'REPUBLICAN': 'MICHAEL PENCE'}},
                      '2020': {'elected_president': 'JOE BIDEN',
                               'elected_vice_president': 'KAMALA HARRIS',
                               'electoral_votes': {'DEMOCRAT': 306, 'REPUBLICAN': 232},
                               'popular_votes': {'DEMOCRAT': 81300000, 'REPUBLICAN': 74200000},
                               'president_candidates': {'DEMOCRAT': 'JOE BIDEN',
                                                        'REPUBLICAN': 'DONALD TRUMP'},
                               'vice_president_candidates': {'DEMOCRAT': 'KAMALA HARRIS',
                                                             'REPUBLICAN': 'MICHAEL PENCE'}}}

# # # round popular votes to 100 000 and then prety print
# # for year, value in year_to_presidents.items():
# #     for party in value['popular_votes'].keys():
# #         value['popular_votes'][party] = int(
# #             round(value['popular_votes'][party] / 100000) * 100000)

# #     import pprint

# # pprint.pprint(year_to_presidents)
# # # print(year_to_presidents)

# # exit()

# for year, value in sorted(year_to_presidents.items()):
#     parties = dict()
#     for key in value.keys():
#         for party in value[key]:
#             if isinstance(value[key], dict):
#                 parties[party] = parties.get(party, dict())
#                 parties[party][key] = value[key][party]

#     print(f'{year}:\n')
#     for party in sorted(parties.keys(), reverse=True):
#         print(f'\t{party}\t', end='')
#         for key in sorted(parties[party].keys()):
#             print(f'{key}: {parties[party][key]}', end='\t')
#         print()
#     print()
