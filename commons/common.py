import math

import collections
import functools

from difflib import SequenceMatcher

import glob

import copy
import json
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from enum import Enum, auto, IntEnum
import pickle
from urllib import request

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModel
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import bitsandbytes as bnb
from urllib.parse import urlparse
import ssl
import os
from datetime import datetime
import warnings
import logging.config
from collections.abc import Iterable
import torch
from importlib import import_module
from torch import nn as nn
from scipy.stats import rankdata
from bardapi import Bard
import unittest

###
# Logger
logger = logging.getLogger('commons')

use_cuda = torch.cuda.is_available()

###
# Enums
###
# Ref: https://github.com/irgeek/StrEnum
class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    # pylint: disable=no-self-argument
    # The first argument to this function is documented to be the name of the
    # enum member, not `self`:
    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, *_):
        return name


###
# Utility functions

def basename(path, ext=True):
    bn = os.path.basename(path)
    if not ext:
        bn = os.path.splitext(bn)[0]
    return bn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def baseurl(url):
    return re.search(r'://(.+?)/', url).group(1)



###
# Utility classes
###
class StrUtils(object):
    @staticmethod
    def rm_nonwords(s):
        import re
        return re.sub('\\W', '', s)

    @staticmethod
    def empty(s):
        return (pd.isnull(s)) or (len(s) == 0)

    @staticmethod
    def default_str(s, ds=''):
        return ds if pd.isnull(s) else s

    @staticmethod
    def search_digit(s, default=0, last=True):
        the = re.findall("(\d+)", s)
        if the is None or len(the) == 0:
            return default
        else:
            return int(the[-1] if last else the[0])

    @staticmethod
    def sub_between(s, ss, es):
        begin = s.find(ss)
        end = s.find(es, begin)
        if 0 <= begin < end:
            return s[begin+len(ss):end]
        else:
            return None

    @staticmethod
    def extract_integers(s):
        return list(map(int, re.findall(r'\d+', s)))

    @staticmethod
    def extract_floats(s):
        return list(map(float, re.findall(r'\d+\.\d+', s)))

    @staticmethod
    def similarity(s1, s2):
        return SequenceMatcher(None, s1, s2).ratio()

    @classmethod
    def format_keys(cls, fmt):
        return re.findall(r'\{(.*?)\}', fmt)

class RemoteUtils(object):
    _ssl_context = ssl._create_unverified_context()

    @classmethod
    def download_to(cls, url, decode='utf-8', fnout=None):
        with request.urlopen(url, context=cls._ssl_context) as response, open(fnout, 'w') as fout:
            fout.write(response.read().decode(decode))

    @classmethod
    def read_from_url(cls, url, decode='utf-8'):
        data = None
        with request.urlopen(url, context=cls._ssl_context) as response:
            data = response.read().decode(decode)
        return data

    @staticmethod
    def is_url(url):
      try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
      except ValueError:
        return False


class TypeUtils(object):
    @staticmethod
    def is_numeric_value(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_collection(x):
        return isinstance(x, Iterable) and not isinstance(x, str)

class Timestamp(object):
    def start(self):
        self._start = datetime.now()

    def end(self):
        self._end = datetime.now()

class FileUtils(object):
    @staticmethod
    def json_load(fn):
        data = None
        with open(fn, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def json_dump(data, fn):
        with open(fn, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def pkl_load(fn):
        data = None
        with open(fn, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def pkl_dump(data, fn):
        with open(fn, 'wb') as f:
            pickle.dump(data, f)

    # @staticmethod
    # def get_last_version(dirpath, basename):
    #     fns = glob.glob(fn_pattern)
    #     if len(fns) == 0:
    #         return None
    #     the = np.argmax([StrUtils.search_digit(fn[len(basename):], default=0, last=True) for fn in fns])
    #     return sorted(fns)[-1]


class NumpyUtils(object):
    @staticmethod
    def align_by_rrank(x, method='dense'):
        if np.all(x == x[0]):
            return x
        r = rankdata(x, method=method)
        rr = rankdata(-x, method=method)
        return x[[np.where(r == cr)[0][0] for cr in rr]]

    @staticmethod
    def to_probs(x, prob_range=(0.001, 0.999)):
        if np.all(x == x[0]):
            return np.full_like(x, 1 / len(x))
        else:
            probs = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
            probs = probs * (prob_range[1] - prob_range[0]) + prob_range[0]
            return (probs / probs.sum(axis=0)).astype(np.float32)

    @staticmethod
    def rand_assign_numbers(N, maxes=None):
        """
        Get randomly assigned numbers with size of len(maxes) that sums to N
        param N: Target sum of the assigned numbers
        param maxes: maximum number of each element
        """
        assign_ns = np.zeros(len(maxes), dtype=np.int32)
        max_sum = np.sum(maxes)
        if max_sum < N:
            warnings.warn('Sum of maximun choices: %s < %s, N=>%s' % (max_sum, N, max_sum))
            N = max_sum

        while N > 0:
            for i in range(len(assign_ns)):
                if maxes[i] > 0:
                    cur_max = maxes[i] - assign_ns[i]
                    if cur_max > 0 and N > 0:
                        cur = np.random.choice(range(1, min(N, cur_max) + 1), 1)[0]
                        assign_ns[i] += cur
                        N -= cur
        return assign_ns


class LambdaLayer(nn.Module):
    def __init__(self, func=None):
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)


class TorchUtils(object):
    bool_available = hasattr(torch, 'bool')

    @staticmethod
    def collection_to(c, device):
        if torch.is_tensor(c):
            return c.to(device)
        else:
            if isinstance(c, dict):
                new_dict = {}
                for k, v in c.items():
                    new_dict[k] = v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device)
                return new_dict
            elif isinstance(c, list):
                return list(map(lambda v: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device), c))

            elif isinstance(c, tuple):
                return tuple(map(lambda v: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device), c))
            elif isinstance(c, set):
                new_set = set()
                for v in c:
                    new_set.add(v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device))
                return new_set
            else:
                raise ValueError('Input is not tensor and unknown collection type: %s' % type(c))

    @staticmethod
    def to_numpy(x, use_cuda=use_cuda):
        return x.detach().cpu().numpy() if use_cuda else x.detach().numpy()

    @staticmethod
    def create_lambda_layer(func):
        return LambdaLayer(func=func)

    @staticmethod
    def init_module_weights(module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def load_state_dict(ckpt_path, prefix=None):
        sd = torch.load(ckpt_path) if use_cuda else torch.load(ckpt_path, map_location=torch.device('cpu'))
        # Pytorch_lighning has the states as the sub dict named 'state_dict'
        if 'state_dict' in sd:
            sd = sd['state_dict']
        if prefix:
            sub_sd = OrderedDict()
            for key, value in sd.items():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, '')
                    sub_sd[new_key] = value
            return sub_sd
        return sd

    @staticmethod
    def equal_state_dict(st1, st2):
        if st1.keys() != st2.keys():
            return False

        for k, v1 in st1.items():
            v2 = st2[k]
            if (not torch.equal(v1, v2)):
                return False
        return True


class CollectionUtils(object):
    @staticmethod
    def update_dict_key_prefix(the_dict, source_prefix, target_prefix):
        new_dict = OrderedDict()
        for k, v in the_dict.items():
            if k.startswith(source_prefix):
                new_dict[k.replace(source_prefix, target_prefix)] = v
            else:
                new_dict[k] = v
        the_dict.clear()
        the_dict.update(new_dict)

    @classmethod
    def recursive_replace_substr(cls, d, ss1, ss2):
        for k, v in d.items():
            if isinstance(v, str) and ss1 in v:
                d[k] = v.replace(ss1, ss2)
            elif isinstance(v, collections.abc.Mapping):
                cls.recursive_replace_substr(v, ss1, ss2)

    @classmethod
    def get_recursively(cls, d, k):
        # print(f'Current dict: {d}')
        if k in d:
            return d[k]
        for v in d.values():
            if isinstance(v, dict):
                found = cls.get_recursively(v, k)
                if found:
                    return found

    # @classmethod
    # def get_recursively(cls, search_dict, field):
    #     """
    #     Takes a dict with nested lists and dicts,
    #     and searches all dicts for a key of the field
    #     provided.
    #     """
    #     fields_found = []
    #
    #     for key, value in search_dict.items():
    #
    #         if key == field:
    #             fields_found.append(value)
    #
    #         elif isinstance(value, dict):
    #             results = cls.get_recursively(value, field)
    #             for result in results:
    #                 fields_found.append(result)
    #
    #         elif isinstance(value, list):
    #             for item in value:
    #                 if isinstance(item, dict):
    #                     more_results = cls.get_recursively(item, field)
    #                     for another_result in more_results:
    #                         fields_found.append(another_result)
    #
    #     return fields_found

class SlurmUtils(object):
    @staticmethod
    def parse_nodelist(s):
        # TODO: How can handle 2-digit node ID, i.e., bdata01
        if re.search(r'^\w+\[.+\]', s) is None: # single node
            return [s]

        prefix, hnames = tuple(s.split('[', 1))
        hnames = hnames.split(']')[0]
        surfixes = []
        for part in hnames.split(','):
            if '-' in part:
                sa, sb = part.split('-')
                ia, ib = int(sa), int(sb)
                ndigits = min(len(sa), len(sb))
                surfixes.extend(map(lambda i: str(i).zfill(ndigits), range(ia, ib + 1)))
            else:
                # a = int(part)
                surfixes.append(part.strip())
        return ['%s%s' % (prefix, surfix) for surfix in surfixes]

    @classmethod
    def is_slurm_managed(cls):
        return ('SLURM_NTASKS' in os.environ) and (os.environ.get('SLURM_JOB_NAME') != 'bash')

class BardUtils(object):
    bard = None
    token = 'fQjDKVfeCoCqTNFV8zoQkww0w1t_ZrVG4KOT85FlvKPxK2PHpmhhfXeJ7j87Mx_oelNu_A.'

    @classmethod
    def answer(cls, question=None):
        cls.load_bard()
        return cls.bard.get_answer(question)['content']

    @classmethod
    def load_bard(cls):
        if cls.bard is None:
            cls.bard = Bard(token=cls.token)


# class PubmedUtils(object):
#     article_map = {}
#
#     @classmethod
#     def get_article(cls, pmid=None):
#         if pmid not in cls.article_map:
#             cls.article_map[pmid] = PubMedFetcher().article_by_pmid(pmid)
#         return cls.article_map[pmid]


class EntrezUtils(object):
    class DB(StrEnum):
        pubmed = auto()
        protein = auto()
        nucleotide = auto()
        gene = auto()
        taxonomy = auto()
        bioproject = auto()
        biosample = auto()
        sra = auto()
        mesh = auto()
        journal = auto()
        pmc = auto()

    class RetMode(StrEnum):
        fasta = auto()
        xml = auto()
        json = auto()

    BASE_URL_EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    BASE_URL_ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'

    db_title_key_map = {
        DB.pubmed: 'title',
        DB.bioproject: 'project_title'
    }

    @classmethod
    def format_db_id(cls, db=DB.pubmed, id=None):
        return f'{db}:{id}'

    @classmethod
    def parse_db_id(cls, db_id=None):
        tokens = db_id.split(':')
        return tokens[0], tokens[1]

    @classmethod
    def fetch_title(cls, db=DB.pubmed, id=None):
        data = cls.fetch_summary(db=db, id=id, retmode=cls.RetMode.json)
        if db not in cls.db_title_key_map:
            raise ValueError(f'Unsupported db: {db}')
        title_key = cls.db_title_key_map[db]
        return CollectionUtils.get_recursively(data, title_key)

    @classmethod
    def fetch_summary(cls, db=DB.pubmed, id=None, retmode=RetMode.json):
        url = f'{cls.BASE_URL_EFETCH}?db={db}&id={id}&rettype=docsum&retmode={retmode}'
        logger.debug(f'Fetching summary from {url}')
        data = RemoteUtils.read_from_url(url)
        if retmode == cls.RetMode.json:
            data = eval(data)
        return data


E_DB = EntrezUtils.DB
E_RM = EntrezUtils.RetMode


class ReflectionUtils(object):
    @classmethod
    def get_class_name(cls, obj):
        return obj.__class__.__name__

    @classmethod
    def create_instance(cls, class_path, *args, **kwargs):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        module_path, class_name = cls.split_module_class(class_path)
        module = import_module(module_path)
        if '.' in class_name:
            return functools.reduce(_getattr, [module] + class_name.split('.'))(*args, **kwargs)
        else:
            return getattr(module, class_name)(*args, **kwargs)

    @classmethod
    def split_module_class(cls, class_path):
        if cls.is_module(class_path):
            return class_path, None

        tokens = class_path.split('.')
        for i in range(len(tokens) - 1, 0, -1):
            module_path = '.'.join(tokens[:i])
            class_name = '.'.join(tokens[i:])
            if cls.is_module(module_path):
                return module_path, class_name

        return None, class_path

    @classmethod
    def is_module(cls, module_path):
        try:
            import_module(module_path)
            return True
        except ImportError:
            return False


class ModelingUtils(object):
    AUTO_MODEL_TYPE_MAP = {
        'peft': AutoPeftModel,
        'mlm': AutoModelForMaskedLM,
        'causal_lm': AutoModelForCausalLM
    }

    @classmethod
    def load_peft_lm(cls,
                     lm_type,
                     lm_name_or_path,
                     bits=None,
                     torch_dtype='auto',
                     device_map='auto',
                     force_download=False):
        """
        Load the PEFT lm and tokenizer
        :param lm_type is the type of the model, i.e., 'peft', 'mlm', 'causal_lm'
        :param model_name_or_path is the name or path of the pretrained lm
        :param bits is the number of bits for quantization
        :return: the lm and tokenizer
        """
        def create_bnb_config(bits=None):
            if bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
            elif bits == 8:
                return BitsAndBytesConfig(load_in_8bit=True)
            else:
                return None

        def create_peft_config(modules):
            config = LoraConfig(
                r=16,  # dimension of the updated matrices
                lora_alpha=32,  # parameter for scaling
                target_modules=modules,
                lora_dropout=0.05,  # dropout probability for layers
                bias="none",
                # task_type=TaskType.TOKEN_CLS
            )
            return config

        def find_all_linear_names(model, bits=None):
            cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:  # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        def print_trainable_parameters(model, use_4bit=False):
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                num_params = param.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel
                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            if use_4bit:
                trainable_params /= 2
            logger.info(f"all params: {int(all_param):,d} || trainable params: {int(trainable_params):,d} "
                        f"|| trainable %: {100 * trainable_params / all_param}")

        logger.info(f'Start loading pretrained model from {lm_name_or_path}, '
                    f'bits: {bits}, force_download: {force_download}...')
        model = cls.AUTO_MODEL_TYPE_MAP[lm_type].from_pretrained(lm_name_or_path,
                                                                 quantization_config=create_bnb_config(bits=bits),
                                                                 # load_in_8bit=(bits == 8),
                                                                 # load_in_4bit=(bits == 4),
                                                                 torch_dtype=torch_dtype,
                                                                 device_map=device_map,
                                                                 force_download=force_download)
        # Using the prepare_model_for_kbit_training method from PEFT
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing=model.supports_gradient_checkpointing)

        # Create peft model with lora module names
        modules = find_all_linear_names(model)
        peft_config = create_peft_config(modules)
        logger.info(f'Creating peft model with lora config: {peft_config}')
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # print_trainable_parameters(model, use_4bit=(bits == 4))
        tokenizer = AutoTokenizer.from_pretrained(lm_name_or_path)
        return model, tokenizer


class MaskedLMEncoder(nn.Module):

    def __init__(self, lm_type='peft', mlm_name_or_path=None):
        super().__init__()
        self.mlm = ModelingUtils.AUTO_MODEL_TYPE_MAP[lm_type].from_pretrained(mlm_name_or_path,
                                                                              # device_map='auto',
                                                                              output_hidden_states=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_name_or_path)

    @property
    def device(self):
        return self.mlm.device

    @property
    def hidden_size(self):
        return self.mlm.config.hidden_size

    @torch.no_grad()
    def encode(self, inputs, max_length=15):
        """
        :param inputs: are the list of sequences to be encoded
        :param max_length: is the max length of the tokenized sequences
        """
        encoded = self.tokenizer(inputs,
                                 padding="max_length",
                                 truncation=False,
                                 max_length=max_length,
                                 return_overflowing_tokens=False,
                                 return_tensors="pt").to(self.device)
        # logger.debug(f'seqs: {seqs}, encoded: {encoded}')
        outputs = self.mlm(**encoded)
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def decode(self, hidden_states):
        """
        :param hidden_states: are the hidden states, shape: (batch_size, max_length, hidden_size)
        :return:
        """
        logits = self.mlm.lm_head(hidden_states)
        token_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def empty_encode(self, batch_size, max_length=15):
        return self.encode([''] * batch_size, max_length=max_length)

    def forward(self, inputs, max_length=15):
        return self.encode(inputs, max_length=max_length)


class BaseTest(unittest.TestCase):
    def setUp(self):
        super().setUp()

        warnings.filterwarnings('ignore')
        logging.config.fileConfig('config/logging.conf')

        self.test_synonyms = [
            'BoLA-AW10', 'BoLA-D18.4', 'BoLA-HD6', 'BoLA-JSP.1', 'BoLA-T2C', 'BoLA-T2a',
            'BoLA-T2b', 'ELA-A1', 'Gogo-B*0101', 'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kbm8', 'H-2-Kwm7',
            'H-2-Kd', 'H-2-Kk', 'H-2-Ld', 'H-2-Lq', 'HLA-Cw1', 'HLA-Cw4', 'HLA-E*01:01',
            'HLA-E*01:03', 'Mamu-A*01', 'Mamu-A01', 'Mamu-A*02', 'Mamu-A02', 'Mamu-A*07', 'Mamu-A07',
            'Mamu-A*11', 'Mamu-A11', 'Mamu-A*2201', 'Mamu-A2201',
            'Mamu-A*2601', 'Mamu-A2*0102', 'Mamu-A7*0103', 'Mamu-B*01', 'Mamu-B*03', 'Mamu-B*04',
            'Mamu-B*08', 'Mamu-B*1001', 'Mamu-B*17', 'Mamu-B*3901', 'Mamu-B*52', 'Mamu-B*6601',
            'Mamu-B*8301', 'Mamu-B*8701', 'Patr-A*0101', 'Patr-A*0301', 'Patr-A*0401',
            'Patr-A*0602', 'Patr-A*0701', 'Patr-A*0901', 'Patr-B*0101', 'Patr-B*0901',
            'Patr-B*1301', 'Patr-B*1701', 'Patr-B*2401', 'RT1A', 'SLA-1*0401',
            'SLA-1*0701', 'SLA-2*0401', 'SLA-3*0401'
        ]
        self.target_classI_alleles = [
            'BoLA-1*023:01', 'BoLA-2*012:01', 'BoLA-3*001:01', 'BoLA-3*002:01',
            'BoLA-6*013:01', 'BoLA-6*041:01', 'BoLA-T2c', 'H2-Db', 'H2-Dd',
            'H2-Kb', 'H2-Kd', 'H2-Kk', 'H2-Ld', 'HLA-A*01:01', 'HLA-A*02:01',
            'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:05', 'HLA-A*02:06',
            'HLA-A*02:07', 'HLA-A*02:11', 'HLA-A*02:12', 'HLA-A*02:16',
            'HLA-A*02:17', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01',
            'HLA-A*03:19', 'HLA-A*11:01', 'HLA-A*23:01', 'HLA-A*24:02',
            'HLA-A*24:03', 'HLA-A*25:01', 'HLA-A*26:01', 'HLA-A*26:02',
            'HLA-A*26:03', 'HLA-A*29:02', 'HLA-A*30:01', 'HLA-A*30:02',
            'HLA-A*31:01', 'HLA-A*32:01', 'HLA-A*32:07', 'HLA-A*32:15',
            'HLA-A*33:01', 'HLA-A*66:01', 'HLA-A*68:01', 'HLA-A*68:02',
            'HLA-A*68:23', 'HLA-A*69:01', 'HLA-A*80:01', 'HLA-B*07:02',
            'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*14:01',
            'HLA-B*14:02', 'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03',
            'HLA-B*15:09', 'HLA-B*15:17', 'HLA-B*18:01', 'HLA-B*27:05',
            'HLA-B*27:20', 'HLA-B*35:01', 'HLA-B*35:03', 'HLA-B*37:01',
            'HLA-B*38:01', 'HLA-B*39:01', 'HLA-B*40:01', 'HLA-B*40:02',
            'HLA-B*40:13', 'HLA-B*42:01', 'HLA-B*44:02', 'HLA-B*44:03',
            'HLA-B*45:01', 'HLA-B*46:01', 'HLA-B*48:01', 'HLA-B*51:01',
            'HLA-B*53:01', 'HLA-B*54:01', 'HLA-B*57:01', 'HLA-B*57:03',
            'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*73:01', 'HLA-B*81:01',
            'HLA-B*83:01', 'HLA-C*03:03', 'HLA-C*04:01', 'HLA-C*05:01',
            'HLA-C*06:02', 'HLA-C*07:01', 'HLA-C*07:02', 'HLA-C*08:02',
            'HLA-C*12:03', 'HLA-C*14:02', 'HLA-C*15:02', 'HLA-E*01:01',
            'HLA-E*01:03', 'Mamu-A1*001:01', 'Mamu-A1*002:01', 'Mamu-A1*007:01',
            'Mamu-A1*011:01', 'Mamu-A1*022:01', 'Mamu-A1*026:01',
            'Mamu-A2*01:02', 'Mamu-A7*01:03', 'Mamu-B*001:01', 'Mamu-B*003:01',
            'Mamu-B*008:01', 'Mamu-B*010:01', 'Mamu-B*017:01', 'Mamu-B*039:01',
            'Mamu-B*052:01', 'Mamu-B*066:01', 'Mamu-B*084:01', 'Mamu-B*087:01',
            'Patr-A*01:01', 'Patr-A*03:01', 'Patr-A*04:01', 'Patr-A*07:01',
            'Patr-A*09:01', 'Patr-B*01:01', 'Patr-B*13:01', 'Patr-B*24:01',
            'Rano-A1*b', 'SLA-1*04:01', 'SLA-1*07:01', 'SLA-2*04:01','SLA-3*04:01']

    def assertArrayEqual(self, a1, a2):
        np.testing.assert_array_equal(a1, a2)


class FuncTest(BaseTest):
    def test_basename(self):
        self.assertEqual('test.txt', basename('foo/bar/test.txt', ext=True))
        self.assertEqual('test', basename('foo/bar/test.txt', ext=False))

# class BindLevelTest(BaseTest):
#     def test_int(self):
#         self.assertEqual(0, BindLevel.NEGATIVE)
#         self.assertEqual(1, BindLevel.POSITIVE_LOW)
#         self.assertEqual(2, BindLevel.POSITIVE)
#         self.assertEqual(3, BindLevel.POSITIVE_HIGH)
#
#     def test_is_binder(self):
#         self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE_LOW))
#         self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE))
#         self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE_HIGH))
#         self.assertFalse(BindLevel.is_binder(BindLevel.NEGATIVE))
#
#     def test_str(self):
#         self.assertEqual('POSITIVE_LOW', str(BindLevel.POSITIVE_LOW))
#         self.assertEqual('POSITIVE', str(BindLevel.POSITIVE))
#         self.assertEqual('POSITIVE_HIGH', str(BindLevel.POSITIVE_HIGH))
#         self.assertEqual('NEGATIVE', str(BindLevel.NEGATIVE))
#
#     def test_from_str(self):
#         self.assertEqual(BindLevel.POSITIVE_LOW, BindLevel.from_str('positive_low'))
#         self.assertEqual(BindLevel.POSITIVE, BindLevel.from_str('POSITIVE'))
#         self.assertEqual(BindLevel.POSITIVE, BindLevel.from_str('positive'))
#         self.assertEqual(BindLevel.POSITIVE_HIGH, BindLevel.from_str('POSITIVE_HIGH'))
#         self.assertEqual(BindLevel.POSITIVE_HIGH, BindLevel.from_str('positive_high'))
#
#     def test_from_ic50(self):
#         self.assertEqual(BindLevel.POSITIVE_HIGH, BindLevel.from_ic50(90))
#         self.assertEqual(BindLevel.POSITIVE, BindLevel.from_ic50(499))
#         self.assertEqual(BindLevel.POSITIVE_LOW, BindLevel.from_ic50(4999))
#         self.assertEqual(BindLevel.NEGATIVE, BindLevel.from_ic50(5000))


class StrUtilsTest(BaseTest):
    def test_empty(self):
        self.assertTrue(StrUtils.empty(None))
        self.assertTrue(StrUtils.empty(''))
        self.assertFalse(StrUtils.empty(' '))

    def test_default_str(self):
        self.assertEqual('Kim', StrUtils.default_str('Kim', 'Anon'))
        self.assertEqual('ANON', StrUtils.default_str(None, 'Anon').upper())
        self.assertEqual('', StrUtils.default_str(None).upper())
        self.assertEqual('', StrUtils.default_str(np.nan).upper())

    def test_search_digit(self):
        self.assertEqual(0, StrUtils.search_digit('AAA/BB.ckpt'))
        self.assertEqual(1, StrUtils.search_digit('AAA/BBv1.ckpt'))
        self.assertEqual(2, StrUtils.search_digit('AAA/BB_v02.ckpt'))
        self.assertEqual(12, StrUtils.search_digit('AAA/BB_v012.ckpt'))
        self.assertEqual(120, StrUtils.search_digit('AAA/BB_v120.ckpt'))
        self.assertEqual(120, StrUtils.search_digit('_v120.ckpt'))
        self.assertEqual(0, StrUtils.search_digit('.ckpt'))

        self.assertEqual(3, StrUtils.search_digit('AAA/BB_P3.1_v12.ckpt', last=False))
        self.assertEqual(12, StrUtils.search_digit('AAA/BB_P3.1_v12.ckpt', last=True))
        self.assertEqual(12, StrUtils.search_digit('AAA/BB_P3.1_v12.ckpt', last=True))

    def test_sub_between(self):
        self.assertEqual('A K Y D', StrUtils.sub_between('<cls>A K Y D<eos><eos><eos>LLL', '<cls>', '<eos>'))
        self.assertIsNone(StrUtils.sub_between('A K Y D<eos><eos><eos>LLL', '<cls>', '<eos>'))
        self.assertIsNone(StrUtils.sub_between('<cls>A K Y D', '<cls>', '<eos>'))

    def test_format_keys(self):
        self.assertArrayEqual(['A', 'B', 'C'], StrUtils.format_keys('{A}{B}{C}'))
        self.assertArrayEqual(['A', 'B', 'C'], StrUtils.format_keys('{A}:{B}-{C}'))

class TypeUtilsTest(BaseTest):
    def test_is_numeric(self):
        self.assertTrue(TypeUtils.is_numeric_value(1))
        self.assertTrue(TypeUtils.is_numeric_value(1.1))
        self.assertTrue(TypeUtils.is_numeric_value('1.11'))
        self.assertFalse(TypeUtils.is_numeric_value('x'))
        self.assertFalse(TypeUtils.is_numeric_value('1.11xx'))

    def test_is_collection(self):
        self.assertTrue(TypeUtils.is_collection([]))
        self.assertTrue(TypeUtils.is_collection({}))
        self.assertTrue(TypeUtils.is_collection(list()))
        self.assertTrue(TypeUtils.is_collection(set()))
        self.assertTrue(TypeUtils.is_collection(dict()))
        self.assertTrue(TypeUtils.is_collection(np.array([])))
        self.assertFalse(TypeUtils.is_collection('ABC'))
        self.assertFalse(TypeUtils.is_collection(None))


class RemoteUtilsTest(BaseTest):
    def test_download_to(self):
        fn_test = '../tmp/test.fa'
        if os.path.exists(fn_test):
            os.unlink(fn_test)

        url = 'http://www.google.com'

        RemoteUtils.download_to(url, fnout=fn_test)

        self.assertTrue(os.path.exists(fn_test))
        self.assertTrue(os.path.getsize(fn_test) > 0)
        os.unlink(fn_test)

    def test_is_url(self):
        self.assertTrue(RemoteUtils.is_url('http://google.com'))
        self.assertTrue(RemoteUtils.is_url('http://www.uniprot.org/uniprot/P03176'))
        self.assertTrue(RemoteUtils.is_url('http://ontology.iedb.org'))
        self.assertTrue(RemoteUtils.is_url('http://www.allergen.org'))
        self.assertTrue(RemoteUtils.is_url('http://purl.obolibrary.org'))
        self.assertTrue(RemoteUtils.is_url('ftp://purl.obolibrary.org'))
        self.assertFalse(RemoteUtils.is_url('test.txt'))
        self.assertFalse(RemoteUtils.is_url('../data/IEDB/test.txt'))


class HttpMethod(StrEnum):
    GET = auto()
    HEAD = auto
    POST = auto()
    PUT = auto()
    DELETE = auto()
    CONNECT = auto()
    OPTIONS = auto()
    TRACE = auto()
    PATCH = auto()


class StrEnumTest(BaseTest):

    def test_isinstance_str(self):
        self.assertTrue(isinstance(HttpMethod.GET, str))

    def test_value_isinstance_str(self):
        self.assertTrue(isinstance(HttpMethod.GET.value, str))

    def test_str_builtin(self):
        self.assertTrue(str(HttpMethod.GET) == "GET")
        self.assertTrue(HttpMethod.GET == "GET")


class FileUtilsTest(BaseTest):

    def test_get_last_version(self):
        fns = ['../tmp/test.ckpt', '../tmp/test-v1.ckpt', '../tmp/test-v0.ckpt', '../tmp/test-v3.ckpt']
        self.create_empty_files(fns)

        expected = '../tmp/test-v3.ckpt'
        self.assertEqual(expected, FileUtils.get_last_version('../tmp/test*.ckpt'))
        self.delete_files(fns)

        fns = ['../tmp/test.ckpt']
        self.create_empty_files(fns)

        expected = '../tmp/test.ckpt'
        self.assertEqual(expected, FileUtils.get_last_version('../tmp/test*.ckpt'))
        self.delete_files(fns)

    @staticmethod
    def delete_files(fns):
        for fn in fns:
            os.unlink(fn)

    @staticmethod
    def create_empty_files(fns):
        for fn in fns:
            with open(fn, 'w') as f:
                pass


class NumpyUtilsTest(BaseTest):
    def test_align_by_rrank(self):
        np.testing.assert_array_equal(np.array([4, 3, 2, 1]), NumpyUtils.align_by_rrank(np.array([1, 2, 3, 4])))
        np.testing.assert_array_equal(np.array([2, 4, 1, 7]), NumpyUtils.align_by_rrank(np.array([4, 2, 7, 1])))
        np.testing.assert_array_equal(np.array([2, 1, 0, 0]), NumpyUtils.align_by_rrank(np.array([0, 1, 2, 2])))
        np.testing.assert_array_equal(np.array([2, 1, 1, 1]), NumpyUtils.align_by_rrank(np.array([1, 2, 2, 2])))
        np.testing.assert_array_equal(np.array([2, 2, 1, 1, 1, 2, 2]), NumpyUtils.align_by_rrank(np.array([1, 1, 2, 2, 2, 1, 1])))
        np.testing.assert_array_equal(np.array([2, 2, 2, 2]), NumpyUtils.align_by_rrank(np.array([2, 2, 2, 2])))
        np.testing.assert_array_equal(np.array([0, 0, 0, 0]), NumpyUtils.align_by_rrank(np.array([0, 0, 0, 0])))
        np.testing.assert_array_equal(np.array([1, 1, 2]), NumpyUtils.align_by_rrank(np.array([2, 2, 1])))
        np.testing.assert_array_equal(np.array([0.3, 0.2, 0.5]), NumpyUtils.align_by_rrank(np.array([0.3, 0.5, 0.2])))
        np.testing.assert_array_equal(np.array([0.2, 0.2, 0.2, 0.3, 0.1]), NumpyUtils.align_by_rrank(np.array([0.2, 0.2, 0.2, 0.1, 0.3])))

    def test_to_probs(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        probs = NumpyUtils.to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1, delta=1e-5)

        x = np.array([-1, 2, -3, 4, 0], dtype=np.float32)
        probs = NumpyUtils.to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1, delta=1e-5)

        x = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        probs = NumpyUtils.to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1, delta=1e-5)

        x = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        probs = NumpyUtils.to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1, delta=1e-5)

        x = np.array([1, 2, 0, -2, -1], dtype=np.float32)
        probs = NumpyUtils.to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1, delta=1e-5)

        x = np.array([0, 0, 0, 1, 0], dtype=np.float32)
        probs = NumpyUtils.to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1, delta=1e-5)

    def test_rand_assigh_numbers(self):
        for N in np.random.randint(10, 1000, 1000):
            maxes = np.random.randint(2, 100, 10)
            print('N: %s, maxes: %s' % (N, maxes))
            N = min(N, sum(maxes))

            assign_ns = NumpyUtils.rand_assign_numbers(N, maxes)
            print('assign_ns: %s' % assign_ns)

            self.assertEqual(len(maxes), len(assign_ns))
            self.assertEqual(N, sum(assign_ns))
            self.assertTrue(all([assign_ns[i] <= maxes[i] for i in range(len(maxes))]))


class TorchUtilsTest(BaseTest):

    def test_collection_to(self):
        dev = torch.device('cpu')

        self.assertTrue(TorchUtils.collection_to(torch.tensor([1, 2, 3]), dev).device == dev)

        tc = TorchUtils.collection_to({'A': [1, 2], 'B': [3]}, dev)
        print(tc)
        self.assertTrue(isinstance(tc, dict))
        self.assertTrue(torch.is_tensor(tc['A']))
        self.assertTrue(tc['A'].device == dev)

        tc = TorchUtils.collection_to([1, 2, 3], dev)
        print(tc)
        self.assertTrue(isinstance(tc, list))
        self.assertTrue(torch.is_tensor(tc[1]))
        self.assertTrue(tc[1].device == dev)

        tc = TorchUtils.collection_to((1, 2, 3), dev)
        print(tc)
        self.assertTrue(isinstance(tc, tuple))
        self.assertTrue(torch.is_tensor(tc[1]))
        self.assertTrue(tc[1].device == dev)

    def test_create_lambda_layer(self):
        layer = TorchUtils.create_lambda_layer(lambda x: x**2)
        out = layer.forward(torch.tensor(2))
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(torch.tensor(4), out)

        layer = TorchUtils.create_lambda_layer(lambda x: x.view(x.size(0), -1))
        out = layer.forward(torch.randn(4, 3, 2))
        self.assertTrue(torch.is_tensor(out))
        print(out)
        self.assertEqual((4, 6), out.shape)

    def _test_load_state_dict(self, ckpt_path, prefix='bert.'):
        sd = TorchUtils.load_state_dict(ckpt_path, prefix=prefix)
        expected_len = 199
        self.assertEqual(expected_len, len(sd))
        n_layers = 12
        self.assertEqual(1, len(list(filter(lambda k: k.startswith('embeddings.word_embeddings'), sd.keys()))))
        self.assertEqual(1, len(list(filter(lambda k: k.startswith('embeddings.position_embeddings'), sd.keys()))))
        self.assertEqual(1, len(list(filter(lambda k: k.startswith('embeddings.token_type_embeddings'), sd.keys()))))
        for i in range(n_layers):
            expected_keys = [
                f'encoder.layer.{i}.attention.self.query.weight',
                f'encoder.layer.{i}.attention.self.key.weight',
                f'encoder.layer.{i}.attention.self.value.weight',
                f'encoder.layer.{i}.attention.output.dense.weight',
                f'encoder.layer.{i}.intermediate.dense.weight',
                f'encoder.layer.{i}.output.dense.weight'
            ]
            self.assertTrue(all([k in sd.keys() for k in expected_keys]))
        self.assertTrue('pooler.dense.weight' in sd.keys())

    def test_load_state_dict(self):
        self._test_load_state_dict(ckpt_path='../config/bert-base/pytorch_model.bin', prefix='bert.')
        self._test_load_state_dict(ckpt_path='../output/exp1/V3/pretrain.3.3/pretrain.3.3.best_model.ckpt', prefix='bert.')


class CollectionUtilsTest(BaseTest):
    def test_update_dict_key_prefix(self):
        source_prefix = 'bert.'
        target_prefix = 'backbone.'
        sd = OrderedDict({
            f'{source_prefix}embeddings.word_embeddings': 0,
            f'{source_prefix}embeddings.position_embeddings': 1.,
            f'{source_prefix}embeddings.token_type_embeddings': 2.
        })
        n_layers = 12
        v = 3
        for i in range(n_layers):
            sd[f'{source_prefix}encoder.layer.{i}.attention.self.query.weight'] = v
            v += 1
            sd[f'{source_prefix}encoder.layer.{i}.attention.self.key.weight'] = v
            v += 1
            sd[f'{source_prefix}encoder.layer.{i}.attention.self.value.weight'] = v
            v += 1
        no_prefix_keys = [f'no_prefix_key{i}' for i in range(3)]
        for i in range(3):
            sd[no_prefix_keys[i]] = v
            v += 1
        old_sd = copy.deepcopy(sd)
        CollectionUtils.update_dict_key_prefix(sd, source_prefix, target_prefix)
        self.assertEqual(len(old_sd), len(sd))
        self.assertNotEqual(old_sd, sd)
        self.assertTrue(all(old_sd[k] == sd[k] for k in no_prefix_keys))
        self.assertTrue(all(old_sd[k.replace(target_prefix, source_prefix)] == sd[k] for k in filter(lambda k: k not in no_prefix_keys, sd.keys())))
        CollectionUtils.update_dict_key_prefix(sd, target_prefix, source_prefix)
        self.assertEqual(sd, old_sd)

    def test_recursive_update_value(self):
        d = {
            'a': 'v1',
            'b': {
                'c': 'vvx',
                'd': {
                    'e': 'vvx',
                    'f': 'v1'
                }
            }
        }
        self.assertEqual('v1', d['a'])
        self.assertEqual('vvx', d['b']['c'])
        self.assertEqual('vvx', d['b']['d']['e'])
        self.assertEqual('v1', d['b']['d']['f'])

        CollectionUtils.recursive_replace_substr(d, 'vv', 'v2')
        self.assertEqual('v1', d['a'])
        self.assertEqual('v2x', d['b']['c'])
        self.assertEqual('v2x', d['b']['d']['e'])
        self.assertEqual('v1', d['b']['d']['f'])

    def test_get_recursively(self):
        d = FileUtils.json_load('../data/test_esummary.json')
        valid_keys = ['uid', 'project_acc', 'project_name', 'project_title', 'project_description']
        invalid_keys = ['xxx', 'kkk']
        for k in valid_keys:
            v = CollectionUtils.get_recursively(d, k)
            self.assertTrue(len(v) > 0)
            print(f'{k}: {v}')
        for k in invalid_keys:
            v = CollectionUtils.get_recursively(d, k)
            self.assertIsNone(v)

class SlurmUtilsTest(BaseTest):
    def test_parse_nodelist(self):
        self.assertListEqual(['gpu0'], SlurmUtils.parse_nodelist('gpu0'))
        self.assertListEqual(['gpu0', 'gpu1'], SlurmUtils.parse_nodelist('gpu[0, 1]'))
        self.assertListEqual(['gpu0', 'gpu1', 'gpu2'], SlurmUtils.parse_nodelist('gpu[0-2]'))
        self.assertListEqual(['gpu0', 'gpu1', 'gpu2', 'gpu3'], SlurmUtils.parse_nodelist('gpu[0, 1-3]'))
        self.assertListEqual(['gpu0', 'gpu2', 'gpu3', 'gpu4', 'gpu5'], SlurmUtils.parse_nodelist('gpu[0,2, 3-5]'))
        self.assertListEqual(['bdata01'], SlurmUtils.parse_nodelist('bdata01'))
        self.assertListEqual(['bdata01', 'bdata02'], SlurmUtils.parse_nodelist('bdata[01-02]'))
        self.assertListEqual(['bdata02', 'bdata03', 'bdata13', 'bdata14', 'bdata15'], SlurmUtils.parse_nodelist('bdata[02-03, 13-14, 15]'))


    def test_is_slurm_managed(self):
        self.assertFalse(SlurmUtils.is_slurm_managed())
        os.environ['SLURM_NTASKS'] = '16'
        os.environ['SLURM_JOB_NAME'] = 'bash'
        self.assertFalse(SlurmUtils.is_slurm_managed())
        os.environ['SLURM_JOB_NAME'] = 'task1'
        self.assertTrue(SlurmUtils.is_slurm_managed())


class BardUtilsTest(BaseTest):
    def test_answer(self):
        answer = BardUtils.answer('What is the meaning of life?')
        print(answer)
        self.assertTrue(len(answer) > 0)


# class PubmedUtilsTest(BaseTest):
#     def setUp(self):
#         super().setUp()
#         self.pmid = '28531284'
#
#     def test_get_article(self):
#         self.assertTrue(len(PubmedUtils.article_map) == 0)
#
#         article1 = PubmedUtils.get_article(self.pmid)
#         print(article1)
#         self.assertTrue(len(article1.title) > 0)
#         self.assertTrue(len(PubmedUtils.article_map) == 1)
#
#         article2 = PubmedUtils.get_article(self.pmid)
#         self.assertTrue(len(article2.title) > 0)
#         self.assertTrue(len(PubmedUtils.article_map) == 1)
#         self.assertEqual(article1, article2)

class EntrezUtilsTest(BaseTest):
    def setUp(self):
        super().setUp()
        logger.setLevel(logging.DEBUG)

        self.project_ids = [
            "PRJNA330606", "PRJNA393498", "PRJNA280417", "PRJNA506151", "PRJNA321261", "PRJNA312319",
            "PRJNA325496", "PRJNA395098", "PRJNA390125", "PRJNA301507", "PRJNA316033", "PRJNA427746",
            "PRJNA79519", "PRJNA325416", "PRJNA203441", "PRJNA229070",  "PRJNA312766",
            "PRJNA79707", "PRJNA298417", "PRJNA297261", "PRJNA385561", "PRJNA491656", "PRJNA318495",
            "PRJNA318421", "PRJNA389805", "PRJNA300878", "PRJNA516984", "PRJNA510967", "PRJNA509233",
            "PRJNA493983", "PRJNA450033", "PRJNA494572", "PRJNA391483", "PRJNA544470", "PRJNA477518",
            "PRJNA356992", "PRJNA422601", "PRJNA273698", "PRJNA214848", "PRJNA577520", "PRJNA377207",
            "PRJNA376866", "PRJNA389678", "PRJNA544699", "PRJNA495603", "PRJNA374973", "PRJNA412739",
            "PRJNA329041", "PRJNA579190", "PRJNA258001", "PRJNA393071", "PRJNA315543", "PRJNA473147",
            "PRJNA517192", "PRJNA577794", "PRJNA633317", "PRJNA516296", "PRJNA436233", "PRJNA429872",
            "PRJEB38339", "PRJNA449605", "PRJNA208615", "PRJNA377124", "PRJNA602827"
        ]
        self.pmids = [
            '28531284', '15448372', '15519708', '9698550', '15585860',
            '10488102', 12842050, 11183781, 7684468, 15634878, 15694996, 15710332
        ]

    def test_fetch_title(self):
        for prj_id in self.project_ids:
            print(f'Fetching data for {prj_id}...')
            title = EntrezUtils.fetch_title(db=E_DB.bioproject, id=prj_id)
            self.assertTrue(len(title) > 0)
            print(f'{prj_id}: {title}')

        for pmid in self.pmids:
            print(f'Fetching data for {pmid}...')
            title = EntrezUtils.fetch_title(db=E_DB.pubmed, id=pmid)
            self.assertTrue(len(title) > 0)
            print(f'{pmid}: {title}')


class ReflectionUtilsTest(BaseTest):
    class Foo(object):
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __eq__(self, other):
            return self.a == other.a and self.b == other.b

    def test_get_class_name(self):
        self.assertEqual('Foo', ReflectionUtils.get_class_name(ReflectionUtilsTest.Foo(1, b=2)))

    def test_split_module_class(self):
        self.assertEqual(('commons.utils', 'ReflectionUtilsTest'), ReflectionUtils.split_module_class('commons.utils.ReflectionUtilsTest'))
        self.assertEqual(('commons.utils', 'ReflectionUtilsTest.Foo'), ReflectionUtils.split_module_class('commons.utils.ReflectionUtilsTest.Foo'))
        self.assertEqual(('commons.utils', 'ReflectionUtilsTest.Foo.Bar'), ReflectionUtils.split_module_class('commons.utils.ReflectionUtilsTest.Foo.Bar'))
        self.assertEqual(('commons.utils', None), ReflectionUtils.split_module_class('commons.utils'))
        self.assertEqual((None, 'ReflectionUtilsTest'), ReflectionUtils.split_module_class('ReflectionUtilsTest'))
        self.assertEqual((None, 'ReflectionUtilsTest.Foo'), ReflectionUtils.split_module_class('ReflectionUtilsTest.Foo'))

    def test_create_instance(self):
        obj = ReflectionUtils.create_instance('commons.utils.ReflectionUtilsTest.Foo', 1, b=2)
        self.assertEqual(obj, ReflectionUtilsTest.Foo(1, b=2))


class MaskedLMEncoderTest(BaseTest):
    def setUp(self):
        super().setUp()

        self.seq_encoder = MaskedLMEncoder(mlm_name_or_path='../pretrained/peft/esm2_t33_650M_UR50D')
        self.text_encoder = MaskedLMEncoder(lm_type='mlm', mlm_name_or_path='minhpqn/bio_roberta-base_pubmed')
        self.seqs = ['AYKTFPPTEPKCASSYSARSYNEQFF', 'KTFPPTEPKCASSYSARSYNEQFF', 'AFLLFLVLICASSSLADYRYEQYF',
                     'FLAFLLFLVCASSSLADYRYEQYF', 'FYLCFLAFLCASSSLADYRYEQYF', 'FYLCFLAFLLCASSSLADYRYEQYF',
                     'IDFYLCFLAFCASSSLADYRYEQYF', 'IELSLIDFYLCASSSLADYRYEQYF', 'LIDFYLCFLCASSSLADYRYEQYF',
                     'LLFLVLIMLCASSSLADYRYEQYF', 'MIELSLIDFYCASSSLADYRYEQYF', 'SLIDFYLCFLCASSSLADYRYEQYF',
                     'YLCFLAFLLCASSSLADYRYEQYF', 'YLNTLTLAVCATSRERAGGGTDTQYF', 'AFLLFLVLICASRSSGQILYEQYF',
                     'FLAFLLFLVCASRSSGQILYEQYF', 'FYLCFLAFLCASRSSGQILYEQYF', 'FYLCFLAFLLCASRSSGQILYEQYF',
                     'IDFYLCFLAFCASRSSGQILYEQYF', 'IELSLIDFYLCASRSSGQILYEQYF', 'LIDFYLCFLCASRSSGQILYEQYF',
                     'LLFLVLIMLCASRSSGQILYEQYF', 'MIELSLIDFYCASRSSGQILYEQYF', 'SLIDFYLCFLCASRSSGQILYEQYF',
                     'YLCFLAFLLCASRSSGQILYEQYF', 'DTDFVNEFYAYCASSADMGYEKLFF', 'NRDVDTDFVNEFYCASSADMGYEKLFF',
                     'FVDGVPFVVCASSFRGRMETQYF', 'FLPRVFSAVCASSYVGSEQFF', 'AFPFTIYSLCASSYSLGSDNTEAFF',
                     'GYINVFAFPFCASSYSLGSDNTEAFF', 'INVFAFPFTICASSYSLGSDNTEAFF', 'MGYINVFAFCASSYSLGSDNTEAFF',
                     'NVFAFPFTICASSYSLGSDNTEAFF', 'NVFAFPFTIYCASSYSLGSDNTEAFF', 'YINVFAFPFCASSYSLGSDNTEAFF',
                     'YLQPRTFLCASSPEDEQFF', 'YLQPRTFLLCASSPEDEQFF', 'YYVGYLQPRTFCASSPEDEQFF',
                     'KLPDDFTGCVCASSQDPLLAGGRFSAYNEQFF', 'AFLLFLVLICASSLTARGLRSPDTQYF', 'FLAFLLFLVCASSLTARGLRSPDTQYF',
                     'FYLCFLAFLCASSLTARGLRSPDTQYF', 'FYLCFLAFLLCASSLTARGLRSPDTQYF', 'IDFYLCFLAFCASSLTARGLRSPDTQYF',
                     'IELSLIDFYLCASSLTARGLRSPDTQYF', 'LIDFYLCFLCASSLTARGLRSPDTQYF', 'LLFLVLIMLCASSLTARGLRSPDTQYF',
                     'MIELSLIDFYCASSLTARGLRSPDTQYF', 'SLIDFYLCFLCASSLTARGLRSPDTQYF']
        self.texts = ['SARS-CoV-2,nucleocapsid phosphoprotein', 'SARS-CoV-2,nucleocapsid phosphoprotein',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-A*03:01,HLA-B*07:02,HLA-B*51:01,HLA-C*07:02,HLA-C*12:03',
                      'SARS-CoV-2,ORF1ab,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF7b,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF1ab,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF1ab,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF1ab,HLA-A*02:01,HLA-B*35:03,HLA-B*44:02,HLA-C*04:01,HLA-C*05:01',
                      'SARS-CoV-2,ORF1ab,HLA-A*02:01,HLA-B*40:01,HLA-B*57:01,HLA-C*03:04,HLA-C*06:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF10,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,surface glycoprotein,HLA-A*02:01,HLA-B*13:02,HLA-B*18:01,HLA-C*06:02,HLA-C*07:01',
                      'SARS-CoV-2,surface glycoprotein,HLA-A*02:01,HLA-B*13:02,HLA-B*18:01,HLA-C*06:02,HLA-C*07:01',
                      'SARS-CoV-2,surface glycoprotein,HLA-A*02:01,HLA-B*13:02,HLA-B*18:01,HLA-C*06:02,HLA-C*07:01',
                      'SARS-CoV-2,surface glycoprotein,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02',
                      'SARS-CoV-2,ORF7b,HLA-A*01:01,HLA-A*02:01,HLA-B*35:02,HLA-B*39:01,HLA-C*04:01,HLA-C*07:02']

        self.max_seq_len = max([len(seq) for seq in self.seqs]) + 2
        self.max_text_len = 128

    def test_encode(self):
        encoded = self.seq_encoder.encode(self.seqs, self.max_seq_len)
        self.assertEqual(encoded.shape, (len(self.seqs), self.max_seq_len, self.seq_encoder.hidden_size))
        encoded = self.text_encoder.encode(self.texts, self.max_text_len)
        self.assertEqual(encoded.shape, (len(self.seqs), self.max_text_len, self.text_encoder.hidden_size))


if __name__ == '__main__':
    unittest.main()
