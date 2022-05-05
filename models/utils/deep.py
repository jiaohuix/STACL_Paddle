
from fastcore.all import patch_to, partial, MethodType
import functools
import paddle
from paddle.nn.initializer import XavierNormal
import paddle.nn as nn
from paddlenlp.transformers import TransformerModel

from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

@patch_to(nn.Layer)
def apply(self, fn, name=""):
    for n, layer in self.named_children():
        nnmame = n if name == "" else name + "." + n
        layer.apply(fn, nnmame)

    fn(self, name)
    return self


def encoder_forward(self, src, src_mask=None, cache=None, alpha=1.0):
    src_mask = _convert_attention_mask(src_mask, src.dtype)
    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    # Add cache for encoder for the usage like UniLM
    if cache is None:
        src = self.self_attn(src, src, src, src_mask)
    else:
        src, incremental_cache = self.self_attn(src, src, src, src_mask, cache)

    src = residual * alpha + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual * alpha + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)
    return src if cache is None else (src, incremental_cache)


def decoder_forward(
    self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None, alpha=1.0
):
    tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
    memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

    residual = tgt
    if self.normalize_before:
        tgt = self.norm1(tgt)
    if cache is None:
        tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
    else:
        tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, cache[0])
    tgt = residual * alpha + self.dropout1(tgt)
    if not self.normalize_before:
        tgt = self.norm1(tgt)

    residual = tgt
    if self.normalize_before:
        tgt = self.norm2(tgt)
    if cache is None:
        tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
    else:
        tgt, static_cache = self.cross_attn(tgt, memory, memory, memory_mask, cache[1])
    tgt = residual * alpha + self.dropout2(tgt)
    if not self.normalize_before:
        tgt = self.norm2(tgt)

    residual = tgt
    if self.normalize_before:
        tgt = self.norm3(tgt)
    tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = residual * alpha + self.dropout3(tgt)
    if not self.normalize_before:
        tgt = self.norm3(tgt)
    return tgt if cache is None else (tgt, (incremental_cache, static_cache))





def decorator(func, gain=1):
    @functools.wraps(func)
    def wrappper(*args, **kwargs):
        fan_in, fan_out = func(*args, **kwargs)
        return fan_in / (gain ** 2), fan_out / (gain ** 2)

    return wrappper

# def deepnorm_init(m, n,N=6,M=6):
#     # N = 12  # encoder layers
#     # M = 6  # decoder layers
#     if "encoder" in n:
#         alpha = 0.81 * ((N ** 4) * M) ** (1 / 16)
#         beta = 0.87 * ((N ** 4) * M) ** -(1 / 16)
#     elif "decoder" in n:
#         alpha = (3 * M) ** (1 / 4)
#         beta = (12 * M) ** -(1 / 4)
#     else:
#         return
#     xavier_normal = XavierNormal()
#     xavier_normal_beta = XavierNormal()
#     xavier_normal_beta._compute_fans = decorator(
#         xavier_normal_beta._compute_fans, gain=beta
#     )
#     if isinstance(m, nn.Linear):
#         if any(x in n for x in ["linear1", "linear2", "v_proj", "out_proj"]):
#             xavier_normal_beta(m.weight)
#         elif any(x in n for x in ["q_proj", "k_proj"]):
#             xavier_normal(m.weight)
#     if isinstance(m, TransformerEncoderLayer) and "encoder" in n:
#         setattr(m, "forward", MethodType(partial(encoder_forward, alpha=alpha), m))
#     elif isinstance(m, TransformerDecoderLayer) and "decoder" in n:
#         setattr(m, "forward", MethodType(partial(decoder_forward, alpha=alpha), m))

def deepnorm_init(m, n,N=6,M=6):
    # N = 12  # encoder layers
    # M = 6  # decoder layers
    if "encoder" in n:
        # alpha = 0.81 * ((N ** 4) * M) ** (1 / 16)
        beta = 0.87 * ((N ** 4) * M) ** -(1 / 16)
    elif "decoder" in n:
        # alpha = (3 * M) ** (1 / 4)
        beta = (12 * M) ** -(1 / 4)
    else:
        return
    xavier_normal = XavierNormal()
    xavier_normal_beta = XavierNormal()
    xavier_normal_beta._compute_fans = decorator(
        xavier_normal_beta._compute_fans, gain=beta
    )
    if isinstance(m, nn.Linear):
        if any(x in n for x in ["linear1", "linear2", "v_proj", "out_proj"]):
            xavier_normal_beta(m.weight)
        elif any(x in n for x in ["q_proj", "k_proj"]):
            xavier_normal(m.weight)
    # if isinstance(m, TransformerEncoderLayer) and "encoder" in n:
    #     setattr(m, "forward", MethodType(partial(encoder_forward, alpha=alpha), m))
    # elif isinstance(m, TransformerDecoderLayer) and "decoder" in n:
    #     setattr(m, "forward", MethodType(partial(decoder_forward, alpha=alpha), m))

if __name__ == '__main__':
    num_encoder_layers=12
    num_decoder_layers=6
    model = TransformerModel(
        src_vocab_size=100,
        trg_vocab_size=1010,
        max_length=1024,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        n_head=8,
        d_model=512,
        d_inner_hid=1024,
        dropout=0.1,
        weight_sharing=False,
    )
    init_fn=partial(deepnorm_init,N=num_encoder_layers,M=num_decoder_layers)
    model.apply(init_fn)
    paddle.set_grad_enabled(False)
    batch_size = 5
    seq_len = 10
    predict = model(
    src_word=paddle.randint(low=3, high=100, shape=[batch_size, seq_len]),
    trg_word=paddle.randint(low=3, high=100, shape=[batch_size, seq_len]))
    print(predict.shape)