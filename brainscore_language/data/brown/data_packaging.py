import logging
import numpy as np
import pandas as pd
import re
import sys
import collections
from pathlib import Path
from tqdm import tqdm

import pickle

from brainio.assemblies import BehavioralAssembly
# from brainscore_core import BehavioralAssembly
# from brainscore_language.utils.s3 import upload_data_assembly

_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""

def upload_brown(register_plugin=False):
    # adapted from
    # https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/naturalStories.py#L15
    # data_path = Path(__file__).parent / 'naturalstories_RTS'
    data_path = Path("~/language/brainscore_language/data") / 'brown'
    data_file = data_path / 'brown_Y.csv'
    # stories_file = data_path / 'all_stories.tok'
    _logger.info(f'Data file: {data_file}.')

    # get data
    data = pd.read_csv(data_file, delimiter=' ')
    data = data[(data['fdur']>=100) & (data['fdur']<=3000)]
    data = data.reset_index(drop=True)

    # get unique word identifier tuples and order in order of stories
    # item_ID = np.array(data['item'])
    # zone_ID = np.array(data['zone'])
    # zpd_lst = list(zip(item_ID, zone_ID))
    # unique_zpd_lst = list(set(zpd_lst))
    # unique_zpd_lst = sorted(unique_zpd_lst, key=lambda tup: (tup[0], tup[1]))
    text_id = np.array(data['text_id'])
    word_number = np.array(data['Word_Number'])
    zpd_lst = list(zip(text_id, word_number))
    unique_zpd_lst = list(set(zpd_lst))
    unique_zpd_lst = sorted(unique_zpd_lst, key=lambda tup: (tup[0], tup[1]))

    # get unique WorkerIds
    # subjects = data.WorkerId.unique()
    subjects = data.subject.unique()
    # ====== create reading_times ======
    r_dim = len(unique_zpd_lst)
    c_dim = len(subjects)

    # default value for a subject's not having an RT for a story/word is NaN
    reading_times = np.empty((r_dim, c_dim))
    reading_times[:] = np.nan
    end_time = {}

    # set row and column indices for reading_times
    r_indices = {unique_zpd_lst[i]: i for i in range(r_dim)}
    c_indices = {subjects[i]: i for i in range(c_dim)}

    # populate meta information dictionary for subjects xarray dimension
    # metaInfo_subjects = {}

    # counter = 0
    # for index, d in tqdm(data.iterrows(), total=len(data), desc='indices'):
    #     r = r_indices[(d['item'], d['zone'])]
    #     c = c_indices[d['WorkerId']]
    #     reading_times[r][c] = d['RT']
    #     key = d['WorkerId']
    #     if key not in metaInfo_subjects:
    #         metaInfo_subjects[key] = (d['correct'], d['WorkTimeInSeconds'])
    #     else:
    #         counter += 1

    # reading_times = np.array(reading_times)

    # counter = 0
    for index, d in tqdm(data.iterrows(), total=len(data), desc='indices'):
        r = r_indices[(d['text_id'], d['Word_Number'])]
        c = c_indices[d['subject']]
        reading_times[r][c] = d['fdur']
        end_time[d['subject']] = d['time']
        # key = d['subject']
        # if key not in metaInfo_subjects:
        #     metaInfo_subjects[key] = (d['correct'], d['WorkTimeInSeconds'])
        # else:
        #     counter += 1

    # for subject, d in data.groupby('subject'):
    #     end_time[subject] = d['time']

    reading_times = np.array(reading_times)
    end_time = np.array(list(end_time.values()))

    # get subjects' metadata
    # correct_meta = [v[0] for v in metaInfo_subjects.values()]
    # WorkTimeInSeconds_meta = [v[1] for v in metaInfo_subjects.values()]

    # get metadata for presentation dimension
    # word_df = pd.read_csv(stories_file, sep='\t')
    # voc_item_ID = np.array(word_df['item'])
    # voc_zone_ID = np.array(word_df['zone'])
    # voc_word = np.array(word_df['word'])

    # table = collections.defaultdict(list)
    # for item, stories1 in word_df.groupby('item'):
    #     for zone, stories2 in stories1.groupby('zone'):
    #         table['item'].append(stories2.item.values[0])
    #         table['zone'].append(stories2.zone.values[0])
    #         table['word'].append(stories2.word.values[0])
    #         sub_data = data.loc[(data.item==item) & (data.zone==zone)]
    #         table['nItem'].append(sub_data.nItem.unique()[0])
    #         table['meanItemRT'].append(sub_data.meanItemRT.unique()[0])
    #         table['sdItemRT'].append(sub_data.sdItemRT.unique()[0])
    #         table['gmeanItemRT'].append(sub_data.gmeanItemRT.unique()[0])
    #         table['gsdItemRT'].append(sub_data.gsdItemRT.unique()[0])

    table = collections.defaultdict(list)
    for text, stories1 in data.groupby('text_id'):
        for word_num, stories2 in stories1.groupby('Word_Number'):
            table['text_id'].append(stories2.text_id.values[0])
            table['Word_Number'].append(stories2.Word_Number.values[0])
            table['word'].append(stories2.word.values[0])
            sub_data = data.loc[(data.text_id==text) & (data.Word_Number==word_num)]
            table['startofsentence'].append(sub_data.startofsentence.unique()[0])
            table['endofsentence'].append(sub_data.endofsentence.unique()[0])
            table['code'].append(sub_data.code.unique()[0])
            table['sentpos'].append(sub_data.sentpos.unique()[0])
            table['sentid'].append(sub_data.sentid.unique()[0])
            table['docid'].append(sub_data.docid.unique()[0])
            table['wlen'].append(sub_data.wlen.unique()[0])
            table['resid'].append(sub_data.resid.unique()[0])

    table = pd.DataFrame(table)
    voc_text_ID = np.array(table['text_id'])
    voc_word_num = np.array(table['Word_Number'])
    voc_word = np.array(table['word'])
    code = np.array(table['code'])
    sentid = np.array(table['sentid'])
    sentpos = np.array(table['sentpos'])
    docid = np.array(table['docid'])
    startofsentence = np.array(table['startofsentence'])
    endofsentence = np.array(table['endofsentence'])
    wlen = np.array(table['wlen'])
    resid = np.array(table['resid'])

    # get sentence_IDs (finds 481 sentences)
    sentence_ID = []
    idx = 1
    for i, elm in enumerate(voc_word):
        sentence_ID.append(idx)
        if elm.endswith((".", "?", "!", ".'", "?'", "!'", ";'")):
            if i + 1 < len(voc_word):
                if not (voc_word[i + 1].islower() or voc_word[i] == "Mr."):
                    idx += 1

    # get IDs of words within a sentence
    word_within_a_sentence_ID = []
    idx = 0
    for i, elm in enumerate(voc_word):
        idx += 1
        word_within_a_sentence_ID.append(idx)
        if elm.endswith((".", "?", "!", ".'", "?'", "!'", ";'")):
            if i + 1 < len(voc_word):
                if not (voc_word[i + 1].islower() or voc_word[i] == "Mr."):
                    idx = 0
            else:
                idx = 0

    # stimulus_ID
    stimulus_ID = list(range(1, len(voc_word) + 1))

    # add word_core that treats e.g. "\This" and "This" as the same words (to split over)
    word_core = [re.sub(r'[^\w\s]', '', word) for word in voc_word]

    # build xarray
    # voc_word = word
    # voc_item_ID = index of story (1-10)
    # voc_zone_ID = index of words within a story
    # sentence_ID = index of words within each sentence
    # stimulus_ID = unique index of word across all stories
    # subjects = WorkerIDs
    # correct_meta = number of correct answers in comprehension questions
    # time and evid (concatenation of other metadata) were excluded from the packaging
    assembly = BehavioralAssembly(reading_times,
                                  dims=('presentation', 'subject'),
                                  coords={'word': ('presentation', voc_word),
                                          'word_core': ('presentation', word_core),
                                          'text_id': ('presentation', voc_text_ID),
                                          'Word_Number': ('presentation', voc_word_num),
                                          'word_within_sentence_id': ('presentation', word_within_a_sentence_ID),
                                          'sentence_id': ('presentation', sentence_ID),
                                          'stimulus_id': ('presentation', stimulus_ID),
                                          'code': ('presentation', code),
                                          'sentid': ('presentation', sentid),
                                          'sentpos': ('presentation', sentpos),
                                          'docid': ('presentation', docid),
                                          'startofsentence': ('presentation', startofsentence),
                                          'endofsentence': ('presentation', endofsentence),
                                          'wlen': ('presentation', wlen),
                                          'resid': ('presentation', resid),
                                          'subject_id': ('subject', subjects),
                                          'time': ('subject', end_time)
                                          })
    
    with open('test.pickle', 'wb') as f:
        pickle.dump(assembly, f)

    if register_plugin:
        return assembly
    # upload
    # upload_data_assembly(assembly,
    #                      assembly_identifier="Futrell2018")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_brown()
