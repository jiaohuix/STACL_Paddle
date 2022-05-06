# -*- coding:utf-8 -*-
import os
import time
import logging
import math
import random
import paddle
import paddle.nn.functional as F
from paddle.optimizer.lr import ReduceOnPlateau,LRScheduler
from paddle import Tensor
import numpy as np
import shutil
from sacremoses import MosesDetruecaser, MosesDetokenizer

def same_seeds(seed=2021):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def save_model(model, optimizer,save_dir,nbest=5):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    paddle.save(model.state_dict(), os.path.join(save_dir, "model.pdparams"))
    paddle.save(optimizer.state_dict(), os.path.join(save_dir, "model.pdopt"))
    ''' save n best and drop old best'''
    if save_dir.find('best')!=-1:
        base_dir=os.path.dirname(save_dir)
        all_names=os.listdir(base_dir)
        ckpt_names=[name for name in all_names if os.path.isdir(os.path.join(base_dir,name)) and name.find('model')!=-1]
        best_names=[name for name in ckpt_names if name.find('best')!=-1]
        best_names=list(sorted(best_names,key=lambda name: float(name.replace('model_best_',''))))
        if len(best_names)>nbest:
            shutil.rmtree(os.path.join(base_dir,best_names[0]))

# def save_model(model, optimizer, save_dir, best_bleu=None):
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#     paddle.save(model.state_dict(), os.path.join(save_dir, "model.pdparams"))
#     optim_state = optimizer.state_dict()
#     if best_bleu: optim_state['LR_Scheduler'].setdefault('best_bleu', best_bleu)  # save best bleu score in optim state
#     paddle.save(optim_state, os.path.join(save_dir, "model.pdopt"))


class NMTMetric(paddle.metric.Metric):
    def __init__(self, name='convs2s'):
        self.smooth_loss = 0
        self.nll_loss = 0
        self.steps = 0
        self.gnorm = 0
        self._name = name

    @paddle.no_grad()
    def update(self, sum_loss, logits, target, sample_size, pad_id, gnorm):
        '''
        :return: current batch loss,nll_loss,ppl
        '''
        loss = sum_loss / sample_size / math.log(2)
        nll_loss, ppl = calc_ppl(logits, target, sample_size, pad_id)
        self.smooth_loss += float(loss)
        self.nll_loss += float(nll_loss)
        self.steps += 1
        self.gnorm += gnorm
        return loss, nll_loss, ppl

    def accumulate(self):
        '''
        :return:accumulate batches loss,nll_loss,ppl
        '''
        avg_loss = self.smooth_loss / self.steps
        avg_nll_loss = self.nll_loss / self.steps
        ppl = pow(2, min(avg_nll_loss, 100.))
        gnorm = self.gnorm / self.steps
        return avg_loss, avg_nll_loss, ppl, gnorm

    def reset(self):
        self.smooth_loss = 0
        self.nll_loss = 0
        self.steps = 0
        self.gnorm = 0

    def name(self):
        """
        Returns metric name
        """
        return self._name


@paddle.no_grad()
def calc_ppl(logits, tgt_tokens, token_num, pad_id, base=2):
    tgt_tokens = tgt_tokens.astype('int64')
    nll = F.cross_entropy(logits, tgt_tokens, reduction='sum', ignore_index=pad_id)  # bsz seq_len 1
    nll_loss = nll / token_num / math.log(2)  # hard ce
    nll_loss = min(nll_loss.item(), 100.)
    ppl = pow(base, nll_loss)
    return nll_loss, ppl


class ReduceOnPlateauWithAnnael(ReduceOnPlateau):
    '''
        Reduce learning rate when ``metrics`` has stopped descending. Models often benefit from reducing the learning rate
    by 2 to 10 times once model performance has no longer improvement.
        [When lr is not updated for force_anneal times,then force shrink the lr by factor.]
    '''

    def __init__(self,
                 learning_rate,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 force_anneal=50,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 epsilon=1e-8,
                 verbose=False,
                 ):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.force_anneal = args.pop('force_anneal')
        super(ReduceOnPlateauWithAnnael, self).__init__(**args)
        self.num_not_updates = 0

    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr', 'num_not_updates'
        ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        # loss must be float, numpy.ndarray or 1-D Tensor with shape [1]
        if isinstance(metrics, (Tensor, np.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                                                                      "should be (1L,), but the current metrics.shape is {}. Maybe that " \
                                                                      "you should call paddle.mean to process it first.".format(
                metrics.shape)
        elif not isinstance(metrics,
                            (int, float, np.float32, np.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray' or 'paddle.Tensor', but receive {}".
                    format(type(metrics)))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metrics, self.best):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:  # 大于【等于】patience，要更新lr，【并要annel清0】
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                self.num_not_updates = 0
                new_lr = max(self.last_lr * self.factor, self.min_lr)
                if self.last_lr - new_lr > self.epsilon:
                    self.last_lr = new_lr
                    if self.verbose:
                        print('Epoch {}: {} set learning rate to {}.'.format(
                            self.last_epoch, self.__class__.__name__,
                            self.last_lr))
            else:  # Update here
                self.num_not_updates += 1
                if self.num_not_updates >= self.force_anneal:
                    self.num_not_updates = 0
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
                    new_lr = max(self.last_lr * self.factor, self.min_lr)
                    if self.last_lr - new_lr > self.epsilon:
                        self.last_lr = new_lr
                        if self.verbose:
                            print('Epoch {}: {} set learning rate to {} because of force anneal.'.format(
                                self.last_epoch, self.__class__.__name__,
                                self.last_lr))


def force_anneal(scheduler: ReduceOnPlateau, anneal: int):
    setattr(scheduler, 'force_anneal', anneal)
    setattr(scheduler, 'num_not_updates', 0)

    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr', 'num_not_updates'
        ]

    setattr(scheduler, 'state_keys', state_keys)

    def step(self, metrics, epoch=None):
        pass

    setattr(scheduler, 'step', step)
    return scheduler


def ExpDecayWithWarmup(warmup_steps, lr_start, lr_peak, lr_decay):
    ''' warmup and exponential decay'''
    # exp_sched = paddle.optimizer.lr.ExponentialDecay(learning_rate=lr_peak, gamma=lr_decay)
    exp_sched = ReduceOnPlateauWithAnnael(learning_rate=lr_peak, factor=lr_decay)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=exp_sched, warmup_steps=warmup_steps,
                                                 start_lr=lr_start, end_lr=lr_peak, verbose=True)
    return scheduler

def get_logger(loggername, save_path='.'):
    # 创建一个logger
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    save_path = save_path

    # 创建一个handler，用于写入日志文件
    log_path = os.path.join(save_path, "logs")  # 指定文件输出路径，注意logs是个文件夹，一定要加上/，不然会导致输出路径错误，把logs变成文件名的一部分了
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    strtime=time.strftime("%m_%d_%H_%M",time.localtime(time.time())) # 注意冒号会保存不了
    logname = os.path.join(log_path, f"{loggername}_{str(strtime)}.log")  # 指定输出的日志文件名
    fh = logging.FileHandler(logname, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
    fh.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s | %(name)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    待修改：添加unk的处理！
    """
    # sum_eos=sum([t for t in seq if t==eos_idx]) # 预测不出eos 0个或最后一个
    # print(f'sum  eos is :{sum_eos}')
    eos_pos = len(seq) - 1  # 初始化eos索引
    for i, idx in enumerate(seq):  # 找eos位置
        if idx == eos_idx:  # 第一个eos的位置即可
            # if i==0: # 如果遇到输出bos（本例为eos），则取后一个eos
            #     print('it is bos ')
            #     continue
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]  # 取bos和eos中间内容
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)  # 控制是否输出bos和eos
    ]
    return seq


def strip_pad(tensor, pad_id):
    return tensor[tensor != pad_id]


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re
        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(' +', ' ', sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


def to_string(
        tokens,
        vocab,
        bpe_symbol=None,
        extra_symbols_to_ignore=None,
        separator=" "):
    extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
    tokens = [int(token) for token in tokens if int(token) not in extra_symbols_to_ignore]  # 去掉extra tokens
    sent = separator.join(
        vocab.to_tokens(tokens)
    )
    return post_process(sent, bpe_symbol)


def sort_file(gen_path="generate.txt", out_path="result.txt"):
    result = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("H-"):
                result.append(line.strip())
    result = sorted(result, key=lambda line: int(line.split("\t")[0].split("-")[1]))
    result = [line.split("\t")[2].strip().replace("\n","")+"\n" for line in result] # 单句出现\n会导致行数不一致！
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write("".join(result))
    print(f"write to file {out_path} success.")


def get_grad_norm(grads):
    norms = paddle.stack([paddle.norm(g, p=2) for g in grads if g is not None])
    gnorm = paddle.norm(norms, p=2)
    return float(gnorm)

def grad_check(model,topk=30):
    names,grads=[],[]
    for n,p in model.named_parameters():
        if p.grad is None:continue
        names.append(n.replace("_layers.",""))
        grads.append(p.grad)
    # grads norm
    norms = paddle.stack([paddle.norm(g, p=2) for g in grads]).numpy()
    from itertools import chain
    norms=list(chain(*norms))
    # sort
    idx=np.argsort(-np.array(norms))
    names=np.array(names)[idx].tolist()[:topk]
    norms=np.array(norms)[idx].tolist()[:topk]
    idx=np.arange(topk).tolist()
    # import matplotlib.pyplot as plt
    # print(idx)
    # print(norms)
    # print(names)
    # plt.bar(idx,norms)
    # plt.xlabel(names)
    # plt.xticks(names)
    # plt.show()

def stream_decode(waitk,idx,word_list,real_read):
    detok = MosesDetokenizer(lang='en')
    detc = MosesDetruecaser()
    if waitk > 0:
        # for wait-k models, wait k words in the beginning，waitk的前面等了k-1个
        word_list = [''] * (waitk - 1) + word_list
    else:
        # for full sentence model, wait until the end # 对于wait full，前面n-1个都是等待
        word_list = [''] * (len(real_read[idx].numpy()) - 1) + word_list

    final_output = []
    real_output = []
    _read = real_read[idx].numpy()
    sent = ''
    bpe_flag = False

    for j in range(max(len(_read), len(word_list))):
        # append number of reads at step j
        r = _read[j] if j < len(_read) else 0
        if r > 0:
            final_output += [''] * (r - 1)

        # append number of writes at step j
        w = word_list[j] if j < len(word_list) else ''
        # w = w.decode('utf-8')
        real_output.append(w)

        _sent = ' '.join(real_output)

        if len(_sent) > 0:
            _sent += ' a'
            _sent = ' '.join(_sent.split())

            _sent = _sent.replace('@@ ', '') # 要不要去掉空格？
            _sent = detok.detokenize(_sent.split())
            _sent = detc.detruecase(_sent)
            _sent = ' '.join(_sent)
            _sent = _sent[:-1].strip()

        incre = _sent[len(sent):]
        # print('_sent0:', _sent)
        sent = _sent
        # print('sent:', sent)

        if r > 0:
            # if there is read, append a word to write
            # final_output.append(w)
            # final_output.append(str.encode(incre))
            final_output.append(incre)
        else:
            # if there is no read, append word to the final write
            if j >= len(word_list):
                break
            # final_output[-1] += b' '+w
            final_output[-1] += incre

    sequence = "\n".join(final_output) + " \n"
    return sequence


class InverseSquareRoot(LRScheduler):
    def __init__(self, warmup_init_lr, warmup_steps, learning_rate=0.1, last_epoch=-1, verbose=False):
        self.learning_rate = learning_rate
        assert self.learning_rate < 1, "learning_rate must greater than 1 when use inverse_sqrt."
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = warmup_steps

        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (learning_rate - warmup_init_lr) / warmup_steps

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = learning_rate * warmup_steps ** 0.5
        super(InverseSquareRoot, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            return self.decay_factor * self.last_epoch ** -0.5


# import matplotlib.pyplot as plt
#
#
# def draw_curve(x, y, title='', xlabel='', ylabel=''):
#     plt.figure()
#     plt.title(title)
#     plt.plot(x, y)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()
#
#
# def noamdecay(step_num):
#     dmodel=512
#     warmup_steps=4000
#     d=dmodel**-0.5
#     a=step_num**-0.5
#     b=step_num*warmup_steps**(-1.5)
#     lr=d*min(a,b)
#     return lr
#


if __name__ == '__main__':
    logger=get_logger(loggername="transformer_base_share_norm_0",save_path=".")
    logger.info("sas")
    # sched = InverseSquareRoot(warmup_init_lr=0.001, warmup_steps=4000, learning_rate=0.5)
    # x = []
    # lrs = []
    # for i in range(100000):
    #     x.append(i)
    #     sched.step()
    #     lr = sched.get_lr()
    #     lrs.append(lr)
    # draw_curve(x, lrs, title='InverseSquareRoot', xlabel='steps', ylabel='lr')
    # x=[i for i in range(1,100000)]
    # lrs=[noamdecay(i) for i in x]
    # draw_curve(x, lrs, title='NoamDecay', xlabel='steps', ylabel='lr')
