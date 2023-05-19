import logging

from brainscore_language import data_registry
# from data_packaging_v2 import upload_natural_stories
from brainscore_language.data.brown.data_packaging import upload_brown
import pickle
# from brainscore_language.data.futrell2018_v2.data_packaging_v2 import ASSEMBLY_V2
# from brainscore_language.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@proceedings{futrell2018natural,
  title={The Natural Stories Corpus},
  author={Futrell, Richard and Gibson, Edward and Tily, Harry J. and Blank, Idan and Vishnevetsky, Anastasia and
          Piantadosi, Steven T. and Fedorenko, Evelina},
  conference={International Conference on Language Resources and Evaluation (LREC)},
  url={http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf},
  year={2018}
}"""

with open('test.pickle', 'rb') as f:
    ASSEMBLY_V2 = pickle.load(f)

def register_plugin():
    ASSEMBLY_V2.attrs['bibtex'] = BIBTEX
    return ASSEMBLY_V2


data_registry['Brown'] = register_plugin
