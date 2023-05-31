from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

DataMask: TypeAlias = torch.Tensor | np.ndarray


def indices_to_barr(idc: np.ndarray, length: int):
    boolean_array = np.zeros(length, dtype=bool)
    boolean_array[idc] = True
    return boolean_array


def masks_from_df(df):
    test_mask = (df.cohort == 1996).to_numpy()
    train_val = df.query("cohort < 1996")
    train_subset = train_val.sample(frac=0.8, random_state=0)
    train_mask = indices_to_barr(train_subset.index, length=df.shape[0])
    val_mask = indices_to_barr(
        np.setdiff1d(train_val.index, train_subset.index), length=df.shape[0]
    )
    return train_mask, val_mask, test_mask


def train_test_from_df(df, X, y):
    train_mask, val_mask, test_mask = masks_from_df(df)
    train_mask[val_mask] = True
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


@dataclass
class DSTCohortMask:
    train: DataMask
    val: DataMask
    test: DataMask

    @classmethod
    def from_df(cls, df, as_torch: bool = False):
        train_mask, val_mask, test_mask = masks_from_df(df)
        if as_torch:
            train_mask = torch.tensor(train_mask, dtype=torch.bool)
            val_mask = torch.tensor(val_mask, dtype=torch.bool)
            test_mask = torch.tensor(test_mask, dtype=torch.bool)
        return cls(train=train_mask, val=val_mask, test=test_mask)


def std_arr(X, exclude_constant: bool = True):
    if exclude_constant:  # Last entry is constant term and trend
        X[:, :-2] = (X[:, :-2] - X[:, :-2].mean(axis=0)[None, :]) / X[:, :-2].std(
            axis=0
        )[None, :]
        return X
    return (X - X.mean(axis=0)[None, :]) / X.std(axis=0)[None, :]


def std_by_masks(X, masks: DSTCohortMask):
    X[masks.train] = std_arr(X[masks.train])
    X[masks.val] = std_arr(X[masks.val])
    X[masks.test] = std_arr(X[masks.test])
    return X
