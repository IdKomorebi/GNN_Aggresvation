# -*- coding: utf-8 -*-
"""DNN82：精确三元组偏置掩码采样器。"""
from __future__ import annotations

import numpy as np


def _from_sizes(
    sizes: np.ndarray, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    masks = np.zeros((len(sizes), n_general), dtype=np.float32)
    for row, size in enumerate(sizes):
        chosen = rng.choice(n_general, size=int(size), replace=False)
        masks[row, chosen] = 1.0
    return masks


def uniform(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    sizes = rng.randint(1, n_general + 1, size=batch)
    return _from_sizes(sizes, n_general, rng)


def triple_only(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _from_sizes(np.full(batch, 3, dtype=int), n_general, rng)


def _mixture(
    batch: int,
    n_general: int,
    rng: np.random.RandomState,
    p_triple: float,
    p_pair: float,
) -> np.ndarray:
    draw = rng.random_sample(batch)
    sizes = rng.randint(1, n_general + 1, size=batch)
    sizes[draw < p_triple] = 3
    pair = (draw >= p_triple) & (draw < p_triple + p_pair)
    sizes[pair] = 2
    return _from_sizes(sizes, n_general, rng)


def triple70_pair20_uniform10(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _mixture(batch, n_general, rng, p_triple=0.70, p_pair=0.20)


def triple90_pair10(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _mixture(batch, n_general, rng, p_triple=0.90, p_pair=0.10)


def triple50_pair40_uniform10(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _mixture(batch, n_general, rng, p_triple=0.50, p_pair=0.40)


def triple40_pair30_uniform30(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _mixture(batch, n_general, rng, p_triple=0.40, p_pair=0.30)


def _local234(
    batch: int,
    n_general: int,
    rng: np.random.RandomState,
    p_pair: float,
    p_triple: float,
    p_four: float,
    p_uniform: float = 0.0,
) -> np.ndarray:
    draw = rng.random_sample(batch)
    sizes = rng.randint(1, n_general + 1, size=batch)
    pair_end = p_pair
    triple_end = pair_end + p_triple
    four_end = triple_end + p_four
    sizes[draw < pair_end] = 2
    sizes[(draw >= pair_end) & (draw < triple_end)] = 3
    sizes[(draw >= triple_end) & (draw < four_end)] = 4
    assert abs(four_end + p_uniform - 1.0) < 1e-9
    return _from_sizes(sizes, n_general, rng)


def local234_20_60_20(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _local234(batch, n_general, rng, 0.20, 0.60, 0.20)


def local234_uniform10(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _local234(
        batch,
        n_general,
        rng,
        p_pair=0.18,
        p_triple=0.54,
        p_four=0.18,
        p_uniform=0.10,
    )


def _grouped(
    base_sampler,
    batch: int,
    n_general: int,
    rng: np.random.RandomState,
    group_size: int,
) -> np.ndarray:
    """一个掩码连续用于若干样本，让单次梯度更聚焦于同一个子集任务。"""
    n_masks = int(np.ceil(batch / group_size))
    masks = base_sampler(n_masks, n_general, rng)
    return np.repeat(masks, group_size, axis=0)[:batch]


def triple70_group8(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _grouped(
        triple70_pair20_uniform10, batch, n_general, rng, group_size=8
    )


def triple70_group32(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _grouped(
        triple70_pair20_uniform10, batch, n_general, rng, group_size=32
    )


def local234_group8(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _grouped(
        local234_uniform10, batch, n_general, rng, group_size=8
    )


def local234_group32(
    batch: int, n_general: int, rng: np.random.RandomState
) -> np.ndarray:
    return _grouped(
        local234_uniform10, batch, n_general, rng, group_size=32
    )


SAMPLERS = {
    "triple_only": triple_only,
    "triple90_pair10": triple90_pair10,
    "triple70_pair20_uniform10": triple70_pair20_uniform10,
    "triple50_pair40_uniform10": triple50_pair40_uniform10,
    "triple40_pair30_uniform30": triple40_pair30_uniform30,
    "local234_20_60_20": local234_20_60_20,
    "local234_uniform10": local234_uniform10,
    "warm_triple70": triple70_pair20_uniform10,
    "warm_local234": local234_uniform10,
    "triple70_group8": triple70_group8,
    "triple70_group32": triple70_group32,
    "local234_group8": local234_group8,
    "local234_group32": local234_group32,
}


def expected_size_mass(scheme: str, n_general: int = 44) -> dict[int, float]:
    """返回设计分布的精确 size 概率，不为元数据额外生成海量掩码。"""
    focus = {
        "triple_only": {3: 1.00},
        "triple90_pair10": {2: 0.10, 3: 0.90},
        "triple70_pair20_uniform10": {2: 0.20, 3: 0.70},
        "triple50_pair40_uniform10": {2: 0.40, 3: 0.50},
        "triple40_pair30_uniform30": {2: 0.30, 3: 0.40},
        "local234_20_60_20": {2: 0.20, 3: 0.60, 4: 0.20},
        "local234_uniform10": {2: 0.18, 3: 0.54, 4: 0.18},
        "warm_triple70": {2: 0.20, 3: 0.70},
        "warm_local234": {2: 0.18, 3: 0.54, 4: 0.18},
        "triple70_group8": {2: 0.20, 3: 0.70},
        "triple70_group32": {2: 0.20, 3: 0.70},
        "local234_group8": {2: 0.18, 3: 0.54, 4: 0.18},
        "local234_group32": {2: 0.18, 3: 0.54, 4: 0.18},
    }[scheme]
    uniform_mass = {
        "triple70_pair20_uniform10": 0.10,
        "triple50_pair40_uniform10": 0.10,
        "triple40_pair30_uniform30": 0.30,
        "local234_uniform10": 0.10,
        "warm_triple70": 0.10,
        "warm_local234": 0.10,
        "triple70_group8": 0.10,
        "triple70_group32": 0.10,
        "local234_group8": 0.10,
        "local234_group32": 0.10,
    }.get(scheme, 0.0)
    result = {size: uniform_mass / n_general for size in range(1, n_general + 1)}
    for size, probability in focus.items():
        result[size] += probability
    return {size: probability for size, probability in result.items() if probability}
