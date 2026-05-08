"""Plotting helpers for TQIP exploratory analysis."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_patient_count_by_year(df: pd.DataFrame, year_col: str = "YODISCH") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5))
    year_counts = df[year_col].dropna().astype(int).value_counts().sort_index()
    year_counts.plot(kind="bar", ax=ax)
    ax.set_title("TQIP Patient Count by Discharge Year")
    ax.set_xlabel("Discharge Year")
    ax.set_ylabel("Patient Count")
    fig.tight_layout()
    return fig, ax


def plot_withdrawal_lst_by_year(
    df: pd.DataFrame, year_col: str = "YODISCH", value_col: str = "WITHDRAWALLST"
) -> tuple[plt.Figure, np.ndarray]:
    years = sorted(df[year_col].dropna().astype(int).unique())
    if not years:
        years = [2017, 2018, 2019, 2020, 2021, 2022]
    fig, axes = plt.subplots(len(years), 1, sharex=True, figsize=(6, 8), dpi=120, squeeze=False)
    axes = axes.ravel()
    unique_vals = sorted(df[value_col].dropna().unique())
    plot_bins = max(len(unique_vals), 3)
    x_ticks = unique_vals if unique_vals else [0, 1, 2]
    for i, yr in enumerate(years):
        year_slice = df.loc[df[year_col] == yr, value_col]
        sns.histplot(x=year_slice, bins=plot_bins, ax=axes[i], discrete=True)
        axes[i].set_title(str(yr), fontsize=8)
        axes[i].set_xticks(x_ticks)
    fig.suptitle("TQIP Withdrawal of LST Values by Year")
    fig.tight_layout()
    return fig, axes


def plot_temperature_distribution(df: pd.DataFrame, column: str = "TEMPERATURE") -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(5, 2), dpi=120)
    sns.histplot(data=df, x=column, bins=200, ax=axes[0])
    sns.histplot(data=df, x=column, bins=200, ax=axes[1])
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    return fig, axes


def _plot_feature_distributions(
    df: pd.DataFrame, features: list[str], categorical: bool
) -> tuple[plt.Figure, np.ndarray]:
    if not features:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.axis("off")
        return fig, np.array([ax])

    n_cols = 6
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3, n_rows * 2.5)))
    axes = np.array(axes).ravel()

    for i, feat in enumerate(features):
        if categorical:
            sns.histplot(data=df, x=feat, shrink=5, ax=axes[i], discrete=True)
        else:
            sns.histplot(data=df, x=feat, shrink=5, ax=axes[i])
        axes[i].tick_params(axis="both", which="both", labelsize=8)
        axes[i].set_xlabel(feat, fontsize=8)

    for i in range(len(features), len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    return fig, axes


def plot_numeric_feature_distributions(df: pd.DataFrame, num_features: list[str]) -> tuple[plt.Figure, np.ndarray]:
    present_features = [feature for feature in num_features if feature in df.columns]
    return _plot_feature_distributions(df, present_features, categorical=False)


def plot_categorical_feature_distributions(df: pd.DataFrame, cat_features: list[str]) -> tuple[plt.Figure, np.ndarray]:
    present_features = [feature for feature in cat_features if feature in df.columns]
    return _plot_feature_distributions(df, present_features, categorical=True)
