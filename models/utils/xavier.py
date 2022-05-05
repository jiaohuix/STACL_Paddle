import functools
import paddle.nn.initializer as I

def decorator(func, gain=1):
    @functools.wraps(func)
    def wrappper(*args, **kwargs):
        fan_in, fan_out = func(*args, **kwargs)
        return fan_in / (gain ** 2), fan_out / (gain ** 2)

    return wrappper

def xavier_uniform_with_gain(tensor,gain):
    xavier_uniform_ = I.XavierUniform()
    xavier_uniform_._compute_fans = decorator(
        xavier_uniform_._compute_fans, gain=gain
    )
    xavier_uniform_(tensor)