import paddle
def masked_fill(x, mask, value):
    return paddle.where(mask, paddle.to_tensor(value, dtype=x.dtype), x)

def gather(x, axis, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if axis < 0:  # 最后一维
        axis = x.ndim + axis
    nd_index = []
    for k in range(x.ndim):
        if k == axis:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * x.ndim
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.arange(x.shape[k], dtype=index.dtype).reshape(reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    paddle_out = paddle.gather_nd(x, paddle.stack(nd_index, axis=-1)).reshape(index_shape)
    return paddle_out

def scatter(tensor,dim,index,src):
    assert dim==0 or dim==1
    assert tensor.ndim==index.ndim==src.ndim==2
    index=paddle.cast(index,dtype='int64')
    i, j = index.shape
    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    if dim==0:
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    else:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
    # PaddlePaddle updates 的 shape 大小必须与 index 对应
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(src, index=updates_index)
    res=paddle.scatter_nd_add(tensor, index, updates)
    return res
