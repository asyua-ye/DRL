import torch
import numpy as np
import torch.distributions as td




def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def get_mode(dist):
    """Get the (transformed) mode of the distribution.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1134
    """
    if isinstance(dist, td.categorical.Categorical):
        mode = torch.argmax(dist.logits, -1)
    elif isinstance(dist, td.normal.Normal):
        mode = dist.mean
    elif isinstance(dist, td.Independent):
        mode = get_mode(dist.base_dist)
    elif isinstance(dist, td.TransformedDistribution):
        base_mode = get_mode(dist.base_dist)
        mode = base_mode
        for transform in dist.transforms:
            mode = transform(mode)
    return mode



def epsilon_greedy_sample(dist, eps=0.1):
    """Generate greedy sample that maximizes the probability.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1106
    """

    def greedy_fn(dist):
        greedy_action = get_mode(dist)
        if eps == 0.0:
            return greedy_action
        sample_action = dist.sample()
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    if eps >= 1.0:
        return dist.sample()
    else:
        return greedy_fn(dist)


