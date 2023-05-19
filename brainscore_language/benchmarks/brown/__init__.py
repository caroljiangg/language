from brainscore_language import benchmark_registry
from .benchmark import BrownPearsonr

benchmark_registry['Brown-pearsonr'] = BrownPearsonr
