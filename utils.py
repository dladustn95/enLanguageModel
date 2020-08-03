# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

def get_datasetEn(bert_tokenizer, gpt_tokenizer, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList_train = []
    targetList_train = []
    attentionList_train = []
    sourceList_valid = []
    targetList_valid = []
    attentionList_valid = []

    srclines = read(dataset_path + "_train.txt")
    for line in srclines:
        tmp = line.split("|")
        source = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(tmp[0]))
        sourceList_train.append(source)
        target = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(tmp[1]))
        targetList_train.append(target)

    lines = read(dataset_path + "_train_keyword.txt")
    for line in lines:
        tmp = []
        for i in line.split(" "):
            tmp.append(float(i))
        attentionList_train.append(tmp)


    srclines = read(dataset_path + "_valid.txt")
    for line in srclines:
        tmp = line.split("|")
        source = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(tmp[0]))
        sourceList_valid.append(source)
        target = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(tmp[1]))
        targetList_valid.append(target)

    lines = read(dataset_path + "_valid_keyword.txt")
    for line in lines:
        tmp = []
        for i in line.split(" "):
            tmp.append(float(i))
        attentionList_valid.append(tmp)

    return sourceList_train, targetList_train, attentionList_train, sourceList_valid, targetList_valid, attentionList_valid

def get_datasetEn2(bert_tokenizer, gpt_tokenizer, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList_train = []
    targetList_train = []
    sourceList_valid = []
    targetList_valid = []

    srclines = read(dataset_path + "_train.txt")
    for line in srclines:
        tmp = line.split("|")
        source = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(tmp[0]))
        sourceList_train.append(source)
        target = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(tmp[1]))
        targetList_train.append(target)

    srclines = read(dataset_path + "_valid.txt")
    for line in srclines:
        tmp = line.split("|")
        source = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(tmp[0]))
        sourceList_valid.append(source)
        target = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(tmp[1]))
        targetList_valid.append(target)


    return sourceList_train, targetList_train, sourceList_valid, targetList_valid

def get_test_datasetEN(bert_tokenizer, gpt_tokenizer, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList = []
    targetList = []
    attentionList = []

    lines = read(dataset_path + "_test.txt")
    for line in lines:
        tmp = line.split("|")
        source = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(tmp[0]))
        sourceList.append(source)
        target = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(tmp[1]))
        targetList.append(target)

    lines = read(dataset_path + "_test_keyword.txt")
    for line in lines:
        tmp = []
        for i in line.split(" "):
            tmp.append(float(i))
        attentionList.append(tmp)

    return sourceList, targetList, attentionList

def get_test_datasetEN2(bert_tokenizer, gpt_tokenizer, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList = []
    targetList = []

    lines = read(dataset_path + "_test.txt")
    for line in lines:
        tmp = line.split("|")
        source = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(tmp[0]))
        sourceList.append(source)
        target = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(tmp[1]))
        targetList.append(target)

    return sourceList, targetList

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str, dataset_path: str, use_adapter: bool, keyword_module: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M')
    data = dataset_path.split("/")[-1]
    logdir = os.path.join(
        'runs', data + '_' + current_time + '_' + model_name + '_adapter' + str(use_adapter) + '_keymodule' + keyword_module)
    return logdir