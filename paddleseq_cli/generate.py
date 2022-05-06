# -*- coding: utf-8 -*-
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import math
import logging
import paddle
from paddleseq_cli.config import get_arguments, get_config
from reader import prep_dataset,prep_vocab, prep_loader, create_stream_infer_loader
from paddleseq_cli.utils import sort_file, stream_decode,post_process,same_seeds,to_string
from models import build_model, SequenceGenerator
from paddlenlp.metrics import BLEU

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("paddleseq_cli.generate")


@paddle.no_grad()
def generate(conf):
    same_seeds(seed=conf.seed)
    # file op
    if not os.path.exists(conf.SAVE): os.makedirs(conf.SAVE)
    generate_path = os.path.join(conf.SAVE, conf.generate.generate_path)
    sorted_path = os.path.join(conf.SAVE, conf.generate.sorted_path)
    out_file = open(generate_path, 'w', encoding='utf-8') if conf.generate.generate_path and not conf.generate.stream else sys.stdout # 若是stream则直接打印
    # stream
    stream_out_path = os.path.join(conf.SAVE, conf.generate.generate_path) if conf.generate.stream else ""
    if conf.generate.stream:
        fstream = open(stream_out_path, "w", encoding='utf8')

    logger.info(f'configs:\n{conf}')
    # dataloader
    test_dset=None
    if not conf.generate.stream:
        test_dset = prep_dataset(conf, mode='test')
        test_loader = prep_loader(conf, test_dset, mode='test')
    else:
        test_loader = create_stream_infer_loader(conf)
    src_vocab, tgt_vocab = prep_vocab(conf)
    # model
    logger.info('Loading models...')
    model = build_model(conf, is_test=True)
    model.eval()
    # logger.info(f"model architecture:\n{model}")
    scorer = BLEU()
    generator = SequenceGenerator(model, vocab_size=model.tgt_vocab_size, beam_size=conf.generate.beam_size,search_strategy=conf.generate.search_strategy) # 可以加些参数

    # 1.for batch
    logger.info('Predicting...')
    has_target = conf.data.has_target and not conf.generate.stream # stream无target
    for batch_id, batch_data in enumerate(test_loader):
        if (batch_id+1)%1==0:
            logger.info(f'batch_id:[{batch_id + 1}/{len(test_loader)}]')
        samples_id, src_tokens, tgt_tokens, real_read = None, None, None, None
        if has_target:
            samples_id, src_tokens, tgt_tokens = batch_data
        else:
            if conf.generate.stream:
                samples_id, src_tokens, real_read = batch_data # src_tokens [streams,bsz,stream_len], real_read: [bsz,streams]
            else:
                samples_id, src_tokens = batch_data
        bsz = samples_id.shape[0]
        samples = {'id': samples_id, 'nsentences': bsz,
                   'net_input': {'src_tokens': paddle.cast(src_tokens, dtype='int64')},  # 需要和后面生成的cand_indices类型一致
                   'target': tgt_tokens}
        hypos = generator.generate(samples)
        # 2.for sample
        symbol="subword_nmt" if not conf.generate.remain_bpe else None
        for i, sample_id in enumerate(samples["id"].tolist()):
            if not conf.generate.stream:
                # 解码src和tgt，并打印
                src_text = post_process(sentence=" ".join(src_vocab.to_tokens(test_dset[sample_id][0])),
                                              symbol=symbol)
                print("S-{}\t{}".format(sample_id, src_text), file=out_file)
                if has_target:
                    tgt_text = post_process(sentence=" ".join(tgt_vocab.to_tokens(test_dset[sample_id][1])),
                                                  symbol=symbol)
                    print("T-{}\t{}".format(sample_id, tgt_text), file=out_file)

            # 3.for prediction
            for j, hypo in enumerate(hypos[i][: conf.generate.n_best]):  # 从第i个sample的beam=5个hypo中，取best=1个
                # 3.1对hypo后处理
                extra_symbols_to_ignore = [model.bos_id, model.eos_id, model.pad_id, model.unk_id]
                hypo_str = to_string(hypo["tokens"], tgt_vocab, bpe_symbol=symbol,
                                           extra_symbols_to_ignore=extra_symbols_to_ignore)
                if conf.generate.stream:
                    tokens = [int(token) for token in hypo["tokens"] if
                              int(token) not in extra_symbols_to_ignore]  # 去掉extra tokens
                    word_list = tgt_vocab.to_tokens(tokens)
                    sequence = stream_decode(waitk=conf.waitk, idx=i, word_list=word_list, real_read=real_read)
                    fstream.write(sequence)
                # 3.2 打印信息
                score = (hypo["score"] / math.log(2)).item()
                if not conf.generate.stream:
                    print("H-{}\t{:.4f}\t{}".format(sample_id, score, hypo_str), file=out_file)
                    print(
                        "P-{}\t{}".format(sample_id,
                                          " ".join(
                                              map(lambda x: "{:.4f}".format(x),
                                                  # convert from base e to base 2
                                                  (hypo["positional_scores"] / math.log(2)).tolist(),
                                                  )
                                          ),
                                          ),
                        file=out_file
                    )
                # 3.3 记录得分（hypo target）token分数，是索引的
                if has_target and j == 0:
                    scorer.add_inst(cand=hypo_str.split(), ref_list=[tgt_text.split()])
    if conf.generate.stream:
        fstream.close()
    # 打印最终得分
    if has_target:
        logger.info(f"BlEU Score:{scorer.score() * 100:.4f}")
    if conf.generate.generate_path and conf.generate.sorted_path and not conf.generate.stream:
        sort_file(gen_path=generate_path, out_path=sorted_path)


if __name__ == '__main__':
    args = get_arguments()
    conf = get_config(args)
    generate(conf)
