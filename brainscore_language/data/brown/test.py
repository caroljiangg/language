import numpy as np

from brainscore_language import load_dataset


class TestData:
    def test_shape(self):
        assembly = load_dataset('Brown')

        assert len(assembly['word']) == 7188
        assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation'])
        assert len(set(assembly['text_id'].values)) == 13
        assert len(set(assembly['Word_Number'].values)) == 764
        assert len(set(assembly['subject_id'].values)) == 35
        
        rt = assembly.values[0]
        assert min(rt) >= 100
        assert max(rt) <= 3000
        
        mean_assembly = assembly.mean('subject')
        assert not np.isnan(mean_assembly).any()

        assert assembly.bibtex.startswith('@proceedings')
        
