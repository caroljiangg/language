import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_language import load_benchmark
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language import score


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def __init__(self, reading_times):
            self.reading_times = reading_times

        def digest_text(self, stimuli):
            return {'behavior': BehavioralAssembly(
                self.reading_times,
                coords={'stimulus': ('presentation', stimuli), 'stimulus_id': ('presentation', np.arange(len(stimuli)))},
                dims=['presentation'])}

        def start_behavioral_task(self, task: ArtificialSubject.Task):
            if task != ArtificialSubject.Task.reading_times:
                raise NotImplementedError()

    def test_dummy_bad(self):
        benchmark = load_benchmark('Brown-pearsonr')
        print('1')
        reading_times = RandomState(0).random(7188)
        print('2')
        dummy_model = TestBenchmark.DummyModel(reading_times=reading_times)
        print('3')
        score = benchmark(dummy_model)
        assert abs(score) < 0.05

    def test_exact(self):
        benchmark = load_benchmark('Brown-pearsonr')
        dummy_model = TestBenchmark.DummyModel(reading_times=benchmark.data.mean('subject').values)
        score = benchmark(dummy_model)
        assert score == approx(1)

    def test_ceiling(self):
        benchmark = load_benchmark('Brown-pearsonr')
        ceiling = benchmark.ceiling
        assert 0.7 < ceiling < 1
        # assert ceiling == approx(.858, abs=.0005)
        assert ceiling.raw.median('split') == ceiling
        assert ceiling.uncorrected_consistencies.median('split') < ceiling
