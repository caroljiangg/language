from brainscore_language import benchmark_registry
from .benchmark_v2 import Futrell2018Pearsonr

benchmark_registry['Futrell2018-pearsonr-v2'] = Futrell2018Pearsonr
