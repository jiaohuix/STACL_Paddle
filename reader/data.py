import numpy as np
from functools import partial
from paddle.io import DataLoader,BatchSampler
from paddlenlp.data import Vocab, Pad, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.data.sampler import SamplerHelper
from reader.indexed_dataset import MMapIndexedDataset
from .iter_dataloader import LanguagePairDataset,BufferedDataloader

def replace_punc(line):
    ''' 中文预处理未能将所有标点转为英文标点，而词表无中文标点。'''
    punc_map={
        "，":",",
        "？":"?",
        "；":";",
        "！":"!",
        "：":":",
        "（":"(",
        "）":")",
    }
    for k,v in punc_map.items():
        line=line.replace(k,v)
    return line

def read(src_path, tgt_path, train_num=-1,is_test=False, has_target=False):
    if is_test and not has_target:
        with open(src_path, 'r', encoding='utf-8') as src_f:
            for sample_id, src_line in enumerate(src_f.readlines()):
                src_line = replace_punc(src_line.strip())
                if not src_line:
                    continue
                yield {'id': sample_id, 'src': src_line, 'tgt': ''}
    else:
        with open(src_path, 'r', encoding='utf-8') as src_f, open(tgt_path, 'r', encoding='utf-8') as tgt_f:
            for sample_id, (src_line, tgt_line) in enumerate(zip(src_f.readlines()[:train_num], tgt_f.readlines()[:train_num])):
                src_line, tgt_line = replace_punc(src_line.strip()), tgt_line.strip()
                # if not src_line or not tgt_line:
                #     continue
                yield {'id': sample_id, 'src': src_line, 'tgt': tgt_line}

def read_bin(src_path, tgt_path=None, train_num=-1,is_test=False, has_target=False):
    if not has_target:
        src_data =  MMapIndexedDataset(tgt_path)
        for sample_id, src_line in enumerate(src_data):
            yield [src_line,[],sample_id]

    src_data,tgt_data=MMapIndexedDataset(src_path),MMapIndexedDataset(tgt_path)
    for sample_id,(src_line, tgt_line) in enumerate(zip(src_data, tgt_data)):
            yield [src_line,tgt_line,sample_id]


def merge_pref_lang(pref,is_src, src_lang,tgt_lang,use_binary=False):
    filename=pref
    lang=src_lang.strip() if is_src else tgt_lang.strip()
    # if use_binary:
    #     filename=filename+f".{src_lang.strip()}-{tgt_lang.strip()}"
    filename=f"{filename}.{lang}"
    return filename


def prep_dataset(conf, mode='train'):
    assert mode in ['train', 'dev', 'test']
    data_args = conf.data
    merge_file_fn=partial(merge_pref_lang,src_lang=data_args.src_lang,tgt_lang=data_args.tgt_lang,use_binary=data_args.use_binary)
    if mode == 'train':
        src_path = merge_file_fn(data_args.train_pref, is_src=True)
        tgt_path = merge_file_fn(data_args.train_pref, is_src=False)
    elif mode == 'dev':
        src_path = merge_file_fn(data_args.valid_pref, is_src=True)
        tgt_path = merge_file_fn(data_args.valid_pref, is_src=False)
    else:
        src_path = merge_file_fn(data_args.test_pref, is_src=True)
        tgt_path = merge_file_fn(data_args.test_pref, is_src=False)
    if data_args.lazy_load and mode=="train": # train时可能一次加载不下
        dataset=LanguagePairDataset(src_path=src_path,tgt_path=tgt_path)
    else:
        read_fn=read if not data_args.use_binary else read_bin
        dataset = load_dataset(read_fn, src_path=src_path, tgt_path=tgt_path,train_num=conf.train.train_data_size, is_test=(mode == 'test'),
                               has_target=conf.data.has_target,lazy=False)

    return dataset


def prep_vocab(conf):
    data_args = conf.data
    merge_file_fn=partial(merge_pref_lang,src_lang=data_args.src_lang,tgt_lang=data_args.tgt_lang)
    src_vocab_fpath = merge_file_fn(data_args.vocab_pref,is_src=True)
    tgt_vocab_fpath = merge_file_fn(data_args.vocab_pref,is_src=False)
    src_vocab = Vocab.load_vocabulary(
        src_vocab_fpath,
        bos_token=data_args.special_token[0], # 顺序不能颠倒，默认词表顺序排列
        pad_token=data_args.special_token[1],
        eos_token=data_args.special_token[2],
        unk_token=data_args.special_token[3]
    )
    tgt_vocab = Vocab.load_vocabulary(
        tgt_vocab_fpath,
        bos_token=data_args.special_token[0],
        pad_token=data_args.special_token[1],
        eos_token=data_args.special_token[2],
        unk_token=data_args.special_token[3]
    )
    # 是否把vocab词数pad到factor倍数，可以加速训练
    conf.defrost()
    if data_args.pad_vocab:
        padding_vocab = (
            lambda x: (x + data_args.pad_factor - 1) // data_args.pad_factor * data_args.pad_factor
        )
        conf.model.src_vocab_size = padding_vocab(len(src_vocab))
        conf.model.tgt_vocab_size  = padding_vocab(len(tgt_vocab))
    else:
        conf.model.src_vocab_size = len(src_vocab)
        conf.model.tgt_vocab_size = len(tgt_vocab)
    conf.freeze()
    return src_vocab, tgt_vocab


def convert_samples(sample, src_vocab, tgt_vocab):
    sample_id = sample['id']
    source = sample['src'].split()
    target = sample['tgt'].split()
    source = src_vocab.to_indices(source)
    target = tgt_vocab.to_indices(target)
    return source, target, sample_id # only-src时，target=[]



# 过滤掉长度 ≤min_len或者≥max_len 的数据
def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def batchify(insts, bos_idx, eos_idx, pad_idx, is_test=False, has_target=False):
    """
    Put all padded data needed by training into a list.
    # insts是含batch个元素的list，每个batch含src和tgt,和id元素[([],[]),([],[]),...]
    inst:[src,tgt,id,real_read]
    """
    # ★sort by descending source length
    if not is_test:
        neg_src_len = list(map(lambda inst: -len(inst[0]), insts))
        sorted_src_idx = np.argsort(neg_src_len, kind='mergsort')  # 不能用[::-1]，假设在长度全相等时，会从1-n变成n-1;且默认quicksort不稳定
        insts = np.array(insts)[sorted_src_idx].tolist()
    # bos_idx,eos_idx=0,2
    # pad data to full sentence length
    left_pad = Pad(pad_idx, pad_right=False)
    right_pad = Pad(pad_idx, pad_right=True, dtype='int64')
    src_word = left_pad([inst[0] + [eos_idx] for inst in insts])  # src+</s>
    samples_id = Stack()([inst[2] for inst in insts])
    if not is_test:
        prev_word = right_pad([[bos_idx] + inst[1] for inst in insts])  # <s>+tgt
        tgt_word = np.expand_dims(right_pad([inst[1] + [eos_idx] for inst in insts]),
                                  axis=2)  # lbl+</s> # pad时候加了bos或eos，导致size突变，*bsz倍
        data_inputs = [samples_id, src_word, prev_word, tgt_word]
    # test
    else:
        if not has_target:
            data_inputs = [samples_id, src_word]
        else:
            tgt_word = right_pad([inst[0] for inst in insts])
            data_inputs = [samples_id, src_word, tgt_word]

    return data_inputs

def get_sampler(conf,dataset,mode='train'):
    assert mode in ['train','dev','test']
    if mode!='test':
        args=conf.train
        sampler = SamplerHelper(dataset)
        shuffle_batch=args.shuffle_batch
        if args.sort_type == SortType.GLOBAL:
            src_key = (lambda idx, data_source: len(data_source[idx][0]))
            tgt_key = (lambda idx, data_source: len(data_source[idx][1]))
            # Sort twice
            sampler = sampler.sort(key=tgt_key).sort(key=src_key)
        else:  # pool
            if args.shuffle:
                sampler = sampler.shuffle(seed=conf.seed)
            max_key = (lambda idx, data_source: max(
                len(data_source[idx][0]), len(data_source[idx][1])))
            if args.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=args.pool_size)
        # 输入 idx,length（高）,size（宽）, data_source ,返回新的size，这个size默认是mini batch的句子数，也可以自定义为宽度（最大词数）
        batch_size_fn = lambda idx, count, sofar, data_source: max(sofar, len(data_source[idx][0]),len(data_source[idx][1]))
        batch_sampler = sampler.batch(
            batch_size=args.max_tokens,
            drop_last=False,
            batch_size_fn=batch_size_fn, # 返回当前的size（宽度）
            key=lambda size_so_far, minibatch_len: size_so_far * minibatch_len) # 输入宽高，计算token数，和bsz比较

        if shuffle_batch:
            batch_sampler = batch_sampler.shuffle(seed=conf.seed)
        if mode=='train':
            batch_sampler = batch_sampler.shard()
    else:
        batch_sampler = BatchSampler(dataset, batch_size=conf.generate.infer_bsz, drop_last=False)
    return batch_sampler

def prep_loader(conf, dataset, mode='train'):
    assert mode in ['train', 'dev', 'test']
    data_args, model_args, strategy_args, train_args, gen_args = conf.data, conf.model, conf.learning_strategy, conf.train, conf.generate
    # load vocab
    src_vocab, tgt_vocab = prep_vocab(conf)
    batchify_fn = partial(batchify, bos_idx=model_args.eos_idx, eos_idx=model_args.eos_idx,
                          pad_idx=model_args.pad_idx, is_test=mode == 'test',
                          has_target=data_args.has_target)

    if data_args.lazy_load and mode=="train": # 训练时懒加载，仅加载buffer_size大小进内存
        assert data_args.use_binary==True,"current only support binary data for lazy load." #TODO:支持text格式文本懒加载（目前只支持二进制）
        assert isinstance(dataset,LanguagePairDataset)
        dataloader=BufferedDataloader(src_data=dataset.src_data,
                                      tgt_data=dataset.tgt_data,
                                      buffer_size=train_args.pool_size,
                                      sort_type=train_args.sort_type,
                                      max_tokens=train_args.max_tokens,
                                      seed=conf.seed,
                                      shuffle=True,
                                      batchify_fn=batchify_fn)

        if conf.train.resume and mode == 'train':  # resume应该bool,路径由init来决定
            dataloader.set_epoch(conf.train.last_epoch + 1)
            print(f"----- Resume Training: set sampler's epoch to {conf.train.last_epoch + 1} as a random seed")
    else:
        if not data_args.use_binary:
            trans_fn = partial(convert_samples, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
            dataset = dataset.map(trans_fn, lazy=False)
        if mode != 'test' and not data_args.use_binary:
            filt_fn = partial(min_max_filer, max_len=model_args.max_length)
            dataset = dataset.filter(filt_fn)

        batch_sampler = get_sampler(conf, dataset, mode=mode)
        # if conf.train.resume and mode == 'train':  # resume应该bool,路径由init来决定
        #     batch_sampler.set_epoch(conf.train.last_epoch + 1)
        #     print(f"----- Resume Training: set sampler's epoch to {conf.train.last_epoch + 1} as a random seed")
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            num_workers=train_args.num_workers,
        )


    return dataloader

class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"

