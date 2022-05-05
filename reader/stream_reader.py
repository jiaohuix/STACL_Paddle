import numpy as np
from . import apply_bpe
from functools import partial
from paddle.io import DataLoader,BatchSampler
from paddlenlp.data import Pad, Stack
from paddlenlp.datasets import load_dataset
from .data import replace_punc,merge_pref_lang,prep_vocab

def read_streams(infile,use_stream_bpe=False):
    filt_fn=lambda s: s.replace("@@ ","").replace(" @-@ ","-").replace(" .",".").replace(" &apos;","&apos;").lower() # 中文最好不要用moses tokenizer吧，太难滤了
    src_js = [[]]
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if use_stream_bpe: # bpe after read
                txt = line.strip()
                if len(src_js[-1]) == 0 or txt.startswith(src_js[-1][-1][-1]) is False:
                    src_js[-1].append([])
                src_js[-1][-1].append(txt)
            else: # bpe before read
                txt = line.strip()
                # if len(src_js[-1]) == 0 or txt.startswith(src_js[-1][-1][-1]) is False:
                if len(src_js[-1]) == 0:
                    prefix=""
                else:
                    prefix=filt_fn(src_js[-1][-1][-1])
                    len_prefix = len(prefix)
                    prefix = prefix[:int(len_prefix * 0.7)]
                if len(src_js[-1]) == 0 or (filt_fn(txt)).startswith(prefix) is False:
                    src_js[-1].append([])
                src_js[-1][-1].append(txt)
    return src_js


def preprocess_streams(src_js,bpe_dict_path=None,lang="zh"):
    '''cut,bpe and replace words to "" which not real read  '''
    stream_output = []
    bpe=None
    if bpe_dict_path is not None:
        parser = apply_bpe.create_parser()
        args = parser.parse_args(args=['-c', bpe_dict_path])
        bpe = apply_bpe.BPE(args.codes, args.merges, args.separator, None, args.glossaries)
    for talk in src_js:
        stream_output.append([])
        for stream in talk:  # talk是许多话，stream是流式的一句话
            _stream = []  # 对每句话分词和bpe
            for s in stream:  # 一行
                s = s.strip()
                if lang=="zh":
                    s=replace_punc(s)
                # if lang=="zh":
                #     s = ' '.join(jieba.cut(s))
                if bpe is not None:
                    s = bpe.process_line(s, 0)
                _stream.append(s)

            stream_output[-1].append([])  # 同src_js一样，第二个[]内再套stream，是bpe后的stream
            prev_len = 1
            prev_sent = ''
            for s in _stream[:-1]:  # 不含最后一行
                if len(s.split()) > prev_len and prev_sent != '':  # 仍是同一句话
                    stream_output[-1][-1].append(' '.join(s.split()[:-1]))
                else:  # 新的一句话
                    stream_output[-1][-1].append('')
                prev_sent = s
                prev_len = len(s.split())
            if len(_stream) > 0:  # 最后一行单独放进去
                stream_output[-1][-1].append(_stream[-1])
    # check streams len, prevents errors in languages other than Chinese
    for stream_src,stream_new in zip(src_js[0],stream_output[0]):
        assert len(stream_src) == len(stream_new), "preprocessed stream  len should match raw len."
    return stream_output

def get_src_real_read(stream_output):
    '''get src data and real read'''
    filter_src = []
    real_read = []  # real incremental number of read before reading this line
    for talk in stream_output:  # talk含所有的stream（多行增长的话）
        for stream in talk:
            filter_src.append([])
            real_read.append([0])
            for idx, s in enumerate(stream):  # 第idx个sample，s是一行
                real_read[-1][-1] += 1
                if len(s) > 0:
                    filter_src[-1].append(s)
                    if idx < len(stream) - 1:
                        real_read[-1].append(0)
    return filter_src,real_read


def pad_rearrange_src(data,pad_val=0,dtype='int32'):
    '''
    function:
            pad raw stream data [bsz,streams,streams_len] to [bsz,max_streams,max_streams_len]
        then rearange shape to [max_streams,bsz,max_streams_len]
    example:
        #in this case,bsz=2 max_streams=2,max_streams_len=4
        data=[
            [[1654, 1],[1654, 3420, 1]],
            [[42, 1],[42, 432, 1],[42, 432, 1654, 1]]
        ]
        res=pad_rearrange_src(data)
        # result:
            [[[1654    1    0    0]
              [1654 3420    1    0]]

             [[1654 3420    1    0]
              [  42    1    0    0]]

             [[  42  432    1    0]
              [  42  432 1654    1]]]
    '''
    # constant nums
    bsz = len(data)
    max_streams = max([len(stream) for stream in data])
    max_streams_len = max([
        max([len(row) for row in stream])
        for stream in data
        ])
    # pad streams len first
    res1 = [
        [row + (max_streams_len - len(row)) * [pad_val] for row in stream]
        for stream in data
    ]
    # secondly pad streams
    res2 = [
        [stream[i] if i < len(stream) else stream[-1] for i in range(max_streams)]
        for stream in res1
    ]
    # then rearrange shape to [streams,bsz,stream_len]
    res=np.array(res2,dtype=dtype).reshape([max_streams,bsz,max_streams_len])
    return res

def read_stream_data(filter_src,real_read):
    ''' output: {'id': 0, 'src': ['大家', '大家 晚上 好'], 'real_read': [3, 2]} '''
    for sample_id,(stream,read) in enumerate(zip(filter_src,real_read)):
        yield {"id": sample_id, "src": stream, "real_read": read}

def convert_stream_samples1(sample,src_vocab):
    '''
        input: sample like:{'id': 0, 'src': ['大家', '大家 晚上 好'], 'real_read': [3, 2]}
        output:  (0, [[1654], [1654, 3420, 343]], [3, 2])
    '''
    sample_id = sample["id"]
    source=[]
    for line in sample["src"]:
        words=line.strip().split()
        source.append(src_vocab.to_indices(words))
    real_read=sample["real_read"]
    return sample_id,source,real_read

def batchify_stream(insts,eos_idx, pad_idx):
    ''' do pad and rearrange shape'''
    samples_id=Stack()([inst[0] for inst in insts])
    source=pad_rearrange_src([[row+[eos_idx] for row in inst[1]] for inst in insts],pad_val=pad_idx) # [bsz,streams,stream_len]->[max_streams,bsz,max_stream_len]
    real_read=Pad(pad_val=0)([inst[2] for inst in insts]) # [bsz,streams]
    return samples_id,source,real_read

def create_stream_infer_loader(conf):
    # get stream o
    data_args=conf.data
    merge_file_fn=partial(merge_pref_lang,src_lang=data_args.src_lang,tgt_lang=data_args.tgt_lang)
    stream_infile=merge_file_fn(conf.data.test_pref,is_src=True)
    src_js = read_streams(stream_infile,use_stream_bpe=conf.generate.use_stream_bpe)
    bpe_dict_path = conf.data.src_bpe_path if conf.generate.use_stream_bpe else None
    stream_output = preprocess_streams(src_js,bpe_dict_path=bpe_dict_path , lang=conf.data.src_lang)
    filter_src, real_read = get_src_real_read(stream_output)
    dataset=load_dataset(read_stream_data,filter_src=filter_src,real_read=real_read,lazy=False)
    # vocab
    src_vocab,_=prep_vocab(conf)
    # make sample,
    trans_fn=partial(convert_stream_samples1,src_vocab=src_vocab)
    dataset=dataset.map(trans_fn)
    sampler = BatchSampler(dataset, shuffle=False, batch_size=conf.generate.infer_bsz, drop_last=False)
    # batchify: pad and reshape to [streams,bsz,stream_len]
    batchify_fn = partial(batchify_stream, eos_idx=conf.model.eos_idx, pad_idx=conf.model.pad_idx)
    # dataloader
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader

