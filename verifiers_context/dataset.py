"""
Dataset helpers for multi-turn context-managed RLM episodes.

ContextManagedRLMEnv expects each dataset row to have a ``queries`` field
(list of per-turn query strings).  This module provides adapters for the
common cases:

* ``make_multi_turn_dataset`` — wrap an existing single-turn dataset by
  grouping consecutive rows into multi-turn episodes.
* ``from_query_lists`` — build a dataset directly from lists of query
  sequences.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset


def from_query_lists(
    queries: list[list[str]],
    answers: list[str] | None = None,
    *,
    queries_key: str = "queries",
    extra_fields: dict[str, list[Any]] | None = None,
) -> Dataset:
    """Build a HuggingFace Dataset from lists of query sequences.

    Parameters
    ----------
    queries : list[list[str]]
        Each element is a list of per-turn queries for one episode.
    answers : list[str] | None
        Expected final answers (one per episode).  Optional.
    queries_key : str
        Column name for the query lists (default: ``"queries"``).
    extra_fields : dict[str, list[Any]] | None
        Additional columns to include.

    Returns
    -------
    Dataset
        HuggingFace Dataset ready for ``ContextManagedRLMEnv``.

    Example
    -------
    >>> ds = from_query_lists(
    ...     queries=[
    ...         ["Summarize data.csv", "What is the mean of column price?"],
    ...         ["List all files", "Count lines in the largest file"],
    ...     ],
    ...     answers=["42.5", "1024"],
    ... )
    """
    data: dict[str, list[Any]] = {queries_key: queries}
    if answers is not None:
        if len(answers) != len(queries):
            raise ValueError(
                f"answers length ({len(answers)}) != queries length ({len(queries)})"
            )
        data["answer"] = answers
    if extra_fields:
        for k, v in extra_fields.items():
            if len(v) != len(queries):
                raise ValueError(
                    f"extra_fields[{k!r}] length ({len(v)}) != queries length ({len(queries)})"
                )
            data[k] = v
    return Dataset.from_dict(data)


def make_multi_turn_dataset(
    dataset: Dataset,
    n_turns: int,
    *,
    prompt_column: str = "prompt",
    answer_column: str = "answer",
    queries_key: str = "queries",
    stride: int | None = None,
) -> Dataset:
    """Convert a single-turn dataset into multi-turn episodes.

    Groups ``n_turns`` consecutive rows into one episode.  The queries
    are the prompts from each row; the answer is taken from the last row
    in each group.

    Parameters
    ----------
    dataset : Dataset
        Source dataset with individual queries.
    n_turns : int
        Number of turns per episode.
    prompt_column : str
        Column containing the query string (default: ``"prompt"``).
    answer_column : str
        Column containing the expected answer (default: ``"answer"``).
    queries_key : str
        Output column name for the query lists (default: ``"queries"``).
    stride : int | None
        Step size between groups.  Defaults to ``n_turns`` (no overlap).
        Use ``stride=1`` for a sliding window.

    Returns
    -------
    Dataset
        Multi-turn dataset.
    """
    if stride is None:
        stride = n_turns

    prompts = dataset[prompt_column]
    answers = dataset[answer_column] if answer_column in dataset.column_names else None

    groups_queries: list[list[str]] = []
    groups_answers: list[str] = []

    for start in range(0, len(prompts) - n_turns + 1, stride):
        group = prompts[start : start + n_turns]
        groups_queries.append(list(group))
        if answers is not None:
            groups_answers.append(answers[start + n_turns - 1])

    data: dict[str, list[Any]] = {queries_key: groups_queries}
    if groups_answers:
        data["answer"] = groups_answers

    # Carry over other columns from the last row of each group.
    skip = {prompt_column, answer_column}
    for col in dataset.column_names:
        if col in skip:
            continue
        col_data = dataset[col]
        data[col] = [
            col_data[start + n_turns - 1]
            for start in range(0, len(prompts) - n_turns + 1, stride)
        ]

    return Dataset.from_dict(data)
