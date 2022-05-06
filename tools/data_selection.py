'''
利用domain和nondomain的数据，从nondomian中选择和domain相似的数据：
'''
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import logging
from paddleseq_cli.config import get_config, get_arguments
from paddleseq_cli.utils import to_string
from reader import prep_dataset, prep_loader,prep_vocab
from models import build_model
from tqdm import tqdm
import math
import paddle
import paddle.nn.functional as F

# logger
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("paddleseq_cli.data_selection")



@paddle.no_grad()
def get_ppl(model, pad_idx, batch_data):
    model.eval()
    if hasattr(model, "stream"):  # dont's support stream evaluation on dev set.
        model.stream = False

    (samples_id, src_tokens, prev_tokens, tgt_tokens) = batch_data
    logits = model(src_tokens, prev_tokens)[0]
    sent_nll_loss = paddle.sum(
        F.cross_entropy(logits, tgt_tokens, reduction="none", ignore_index=pad_idx).squeeze([-1]),
        axis=-1)  # [bsz,tgt_tokens] 每个句子得到ppl求和得到[bsz]
    weights = paddle.cast(tgt_tokens != pad_idx, dtype=paddle.get_default_dtype())
    sent_len = paddle.sum(weights.squeeze([-1]), axis=-1)
    avg_nll_loss = (sent_nll_loss / sent_len / math.log(2)).numpy().tolist()
    avg_ppl = [pow(2, min(nll, 100)) for nll in avg_nll_loss]
    return avg_ppl


def select_by_classifier():
    pass


@paddle.no_grad()
def select_by_ppl(conf, general_ckpt, domain_ckpt,result_path):
    out_file = open(result_path, 'w', encoding='utf-8')

    # data
    dataset_train = prep_dataset(conf, mode="dev")
    train_loader = prep_loader(conf, dataset_train, mode="dev")
    src_vocab, tgt_vocab = prep_vocab(conf)

    # model
    model_general = build_model(conf, is_test=False)
    model_domain = build_model(conf, is_test=False)
    model_general.eval()
    model_domain.eval()

    # init from ckpt
    assert os.path.exists(general_ckpt) and os.path.exists(
        domain_ckpt), "general_ckpt and domain_ckpt path should not be None!"
    general_state = paddle.load(os.path.join(general_ckpt, "model.pdparams"))
    domain_state = paddle.load(os.path.join(domain_ckpt, "model.pdparams"))
    model_general.set_dict(general_state)
    print(f"General model  load pretrained weight from:{general_ckpt}!")
    model_domain.set_dict(domain_state)
    print(f"Domian model  load pretrained weight from:{domain_ckpt}!")

    # forward ppl
    with paddle.no_grad():
        for batch_data in tqdm(train_loader):
            ppl_g = get_ppl(model_general, pad_idx=conf.model.pad_idx, batch_data=batch_data)
            ppl_d = get_ppl(model_domain, pad_idx=conf.model.pad_idx, batch_data=batch_data)
            (samples_id, src_tokens, prev_tokens, tgt_tokens) = batch_data
            for batch_idx,(sample_id,g,d) in enumerate(zip(samples_id,ppl_g,ppl_d)):
                ppl_diff=float(d-g)
                cur_src=src_tokens[batch_idx].numpy().tolist()
                cur_tgt=tgt_tokens[batch_idx].squeeze([-1]).numpy().tolist()
                extra_symbols_to_ignore = [model_domain.bos_id, model_domain.eos_id, model_domain.pad_id, model_domain.unk_id]
                src_text = to_string(cur_src, src_vocab, bpe_symbol=None,extra_symbols_to_ignore=extra_symbols_to_ignore)
                tgt_text = to_string(cur_tgt, tgt_vocab, bpe_symbol=None,extra_symbols_to_ignore=extra_symbols_to_ignore)

                print(f"ID:{int(sample_id)}\tDIFF:{ppl_diff}\tSRC:{src_text}\tTGT:{tgt_text}",file=out_file)



def sort_fn():
    pass

def parse_file2df(file):
    import pandas as pd
    cols=["ID","DIFF","SRC","TGT"]
    df=pd.read_csv(file,sep="\t",names=cols)
    split_fn=lambda row: row.split(":",1)[1] # 防止把句子内按照：分开
    for col in cols:
        df[col]=df[col].map(split_fn)
    df["ID"]=df["ID"].astype("int")
    df["DIFF"]=df["DIFF"].astype("float32")
    return df

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.write(''.join(res))
    print(f'write to {file} success.')

def choose_by_topk(df,k=5):
    df=df.drop_duplicates(subset=["SRC","TGT"])
    df["DIFF"]=df["DIFF"].map(lambda x:abs(x)) # 绝对值
    print(df.describe())
    df=df.sort_values(by="DIFF", ascending=True)

    selected_df=df[:k]
    # selected_df.to_csv(f"selected_top{k}.csv",index=False)
    return selected_df

def choose_by_threshold(df,threshold=0.95):
    min_val,max_val=df["DIFF"].min(),df["DIFF"].max()
    df["DIFF"]=1-(df["DIFF"]-min_val)/(max_val-min_val)
    mask=df["DIFF"]>threshold
    selected_df = df[mask]
    selected_df=selected_df.sort_values(by="DIFF", ascending=False)
    # selected_df.to_csv(f"selected_threshold{threshold}.csv",index=False)
    return selected_df

def write_pair(df,src_lang,tgt_lang,res_prefix):
    src_path=f"{res_prefix}.{src_lang}"
    tgt_path=f"{res_prefix}.{tgt_lang}"
    src_txts=[line.strip().replace("\n","")+"\n" for line in df["SRC"]]
    tgt_txts=[line.strip().replace("\n","")+"\n" for line in df["TGT"]]
    write_file(src_txts,file=src_path)
    write_file(tgt_txts,file=tgt_path)

if __name__ == '__main__':
    args = get_arguments()  # 直接传parser出来，还能修改下参数啊！
    conf = get_config(args)
    # general_ckpt, domain_ckpt = "ckpt/present/linear", "ckpt/present/asr_fine"  # 命令行噢
    # general_ckpt, domain_ckpt = "ckpt/present/linear", "ckpt/present/mix_asr"  # 命令行噢
    general_ckpt, domain_ckpt = "ckpt/present/linear", "output/ckpt/asr/model_best_17.745"  # 中英
    # general_ckpt, domain_ckpt = "ckpt", "output/ckpt/un_dev/model_best_58.373"  # 英西
    # select_by_ppl(conf, general_ckpt, domain_ckpt,result_path="res_zhen.txt")
    df=parse_file2df(file="res_zhen.txt")
    print(df.describe())
    df=choose_by_topk(df,k=100000)
    write_pair(df,src_lang="zh",tgt_lang="en",res_prefix="domain_zhen_10w_abs") # 从500w中取20w
    # choose_by_threshold(df)