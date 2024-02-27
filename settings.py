# task names and it's path
import pathlib
import itertools
import logging


class TASK2PATH:
    @staticmethod
    def get_path(name, split):
        src_lang, tgt_task = name.split("-")[0], name.split("-")[1]  # ex) en-es, en-ucca, en-amr ...
        src_path, tgt_path, tree_path = None, None, None

        # define task types
        is_parsing = tgt_task in ["ucca", "amr", "cstp"]
        is_tree2text = src_lang in ["ucca", "amr", "cstp"]
        is_translation_from_eng = (src_lang == "en" and tgt_task in MULTI_LANGS)
        is_translation_to_eng = (tgt_task == "en" and src_lang in MULTI_LANGS)
        is_ner = (tgt_task == "ner")
        is_wsd = (tgt_task == "wsd")

        if is_tree2text:
            src_lang, tgt_task = tgt_task, src_lang

        if is_parsing or is_tree2text:
            src_path = DATA_DIR / tgt_task / f"{src_lang}" / f"{split}" / f"{src_lang}-{tgt_task}.{src_lang}"
            tgt_path = DATA_DIR / tgt_task / f"{src_lang}" / f"{split}" / f"{src_lang}-{tgt_task}.{tgt_task}"

            if is_tree2text:
                src_path, tgt_path = tgt_path, src_path

            if is_parsing and tgt_task == "amr":
                tree_path = DATA_DIR / tgt_task / f"{src_lang}" / f"{split}" / f"{src_lang}-{tgt_task}.pm"

        if is_translation_to_eng:
            src_lang, tgt_task = tgt_task, src_lang

        if is_translation_to_eng or is_translation_from_eng:
            src_path = DATA_DIR / "parallel_corpus"/ f"en-{tgt_task}.txt" / f"{split}" / f"en-{tgt_task}.en"
            tgt_path = DATA_DIR / "parallel_corpus"/ f"en-{tgt_task}.txt" / f"{split}" / f"en-{tgt_task}.{tgt_task}"

            if is_translation_to_eng:
                src_path, tgt_path = tgt_path, src_path

        # different from other tasks, this returns path to "cached dataset" instead of path to raw data
        if is_ner or is_wsd:
            tgt_path = DATA_DIR / tgt_task / f"{src_lang}" / f"{split}" / f"cached_{src_lang}-{tgt_task}.pkl"

        if not any([is_translation_from_eng, is_translation_to_eng, is_parsing, is_tree2text, is_ner, is_wsd]): # if none of the above is true
            raise AssertionError(f"task name {name} is not defined")

        # Note: tree_path is None if tgt_task is not amr / src_path is none if task is wsd or ner
        return src_path, tgt_path, tree_path

MBART_LANG_CODEMAP = {'en': 'en_XX',
                      'it': 'it_IT',
                      'de': 'de_DE',
                      'es': 'es_XX',
                      'zh': 'zh_CN',
                      'fr': 'fr_XX',
                      'fi': 'fi_FI',
                      'ja': 'ja_XX',
                      'ru': 'ru_RU',
                      'ro': 'ro_RO',
                      'tr': 'tr_TR',
                      'hi': 'hi_IN',
                      'he': 'he_IL',
                      'pl': 'pl_PL',
                      'cs': 'cs_CZ',
                      'et': 'et_EE',
                      'id': 'id_ID',
                      'sv': 'sv_SE',
                      'nl': 'nl_XX',
                      'ko': 'ko_KR',
                      "hr": "hr_HR",
                      "fa": "fa_IR",
                      'ucca': 'ucca',
                      'amr': 'amr',
                      'cstp': 'cstp'}  # hi_IN for a debug purpose

PARSING_TASKS = ['amr', 'ucca', "cstp"]
MULTI_LANGS = [lang for lang in MBART_LANG_CODEMAP if lang not in PARSING_TASKS]

token_cls_tasks = ["wsd", "ner"]
# seq2seq task names e.g. en-es, en-ucca, en-amr, en-wsd, en-ner
SEQ2SEQ_TASKS= [f"{task1}-{task2}" for task1, task2 in itertools.permutations(MBART_LANG_CODEMAP, 2)]
# token classifcation task names e.g. en-wsd, en-ner 
TOKEN_CLS_TASKS = [f"{lang}-{task}" for lang in MULTI_LANGS for task in token_cls_tasks]


PROJECT_DIR = pathlib.Path(__file__).parent.resolve() # /maml4amr
DATA_DIR = PROJECT_DIR / "data"

TEMP = PROJECT_DIR / "temp"
AMR_SCRIPT = PROJECT_DIR / "AMR"
LOG_DIR = PROJECT_DIR / "logs"

logger = logging.getLogger()
logfmt = '%(asctime)s - %(levelname)s - \t%(message)s'
logging.basicConfig(format=logfmt, datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)