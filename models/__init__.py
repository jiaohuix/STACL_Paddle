import os
from .simul_transformer import transformer_simul_base,transformer_simul_big,transformer_base_share
from .simul_transformer_deep import transformer_simul_base_norm,transformer_deep_encoder,transformer_base_share_norm,deep_encoder_share
from .seq_generator import SequenceGenerator
from .utils.deep import deepnorm_init
import models

def build_model(conf,is_test=False):
    model_args,gen_args=conf.model,conf.generate
    model_path=os.path.join(model_args.init_from_params,'model.pdparams')
    model_path=None if not os.path.exists(model_path) else model_path
    model=getattr(models,model_args.model_name)(
                                        is_test=is_test,
                                        pretrained_path=model_path,
                                        src_vocab_size=model_args.src_vocab_size,
                                        tgt_vocab_size=model_args.tgt_vocab_size,
                                        waitk=conf.waitk,
                                        max_length=model_args.max_length,
                                        dropout=model_args.dropout,
                                        stream=gen_args.stream,
                                        )
    return model