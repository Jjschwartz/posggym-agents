from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from posggym_agents.agents.registration import parse_policy_id


def add_95CI(df: pd.DataFrame) -> pd.DataFrame:
    """Add 95% CI columns to dataframe."""

    def conf_int(row, prefix):
        std = row[f"{prefix}_std"]
        n = row["num_episodes"]
        return 1.96 * (std / np.sqrt(n))

    prefix = ""
    for col in df.columns:
        if not col.endswith("_std"):
            continue
        prefix = col.replace("_std", "")
        df[f"{prefix}_CI"] = df.apply(lambda row: conf_int(row, prefix), axis=1)
    return df


def add_outcome_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Add proportion columns to dataframe."""

    def prop(row, col_name):
        n = row["num_episodes"]
        total = row[col_name]
        return total / n

    columns = ["num_LOSS", "num_DRAW", "num_WIN", "num_NA"]
    new_column_names = ["prop_LOSS", "prop_DRAW", "prop_WIN", "prop_NA"]
    for col_name, new_name in zip(columns, new_column_names):
        if col_name in df.columns:
            df[new_name] = df.apply(lambda row: prop(row, col_name), axis=1)
    return df


def clean_df_policy_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Remove environment name from policy ID, if it's present."""

    def clean(row):
        if "/" in row["policy_id"]:
            return row["policy_id"].split("/")[1]
        return row["policy_id"]

    df["policy_id"] = df.apply(clean, axis=1)
    return df


def add_df_coplayer_policy_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add co-player policy ID to dataframe."""
    assert len(df["agent_id"].unique().tolist()) == 2

    df_0 = df[df["agent_id"] == 0]
    df_1 = df[df["agent_id"] == 1]
    # disable warning
    pd.options.mode.chained_assignment = None
    df_0["coplayer_policy_id"] = df_0["exp_id"].map(
        df_1.set_index("exp_id")["policy_id"].to_dict()
    )
    df_1["coplayer_policy_id"] = df_1["exp_id"].map(
        df_0.set_index("exp_id")["policy_id"].to_dict()
    )
    # enable warning
    pd.options.mode.chained_assignment = "warn"
    return pd.concat([df_0, df_1]).reset_index(drop=True)


def add_df_multiple_coplayer_policy_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add co-player policy ID to dataframe for envs with > 2 agents.

    Adds a new column for each agent in the environment:

      coplayer_policy_id_0, coplayer_policy_id_1, ..., coplayer_policy_id_N

    Each column contains the policy_id of the agent with corresponding ID for
    the given experiment. This included the row agent so
    if the row["agent_id"] = i
    then row["coplayer_policy_id_i"] = row["policy_id"]

    """
    # disable warning
    pd.options.mode.chained_assignment = None

    agent_ids = df["agent_id"].unique().tolist()
    agent_ids.sort()

    dfs = [df[df["agent_id"] == i] for i in agent_ids]
    for i, df_i in zip(agent_ids, dfs):
        for j, df_j in zip(agent_ids, dfs):
            df_i[f"coplayer_policy_id_{j}"] = df_i["exp_id"].map(
                df_j.set_index("exp_id")["policy_id"].to_dict()
            )
    # enable warning
    pd.options.mode.chained_assignment = "warn"
    return pd.concat(dfs).reset_index(drop=True)


def add_policy_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Add policy seed to dataframe."""

    def policy_seed(row, id_key):
        _, pi_name, _ = parse_policy_id(row[id_key])
        if "_seed" in pi_name:
            seed_token = [t for t in pi_name.split("_") if t.startswith("seed")][0]
            return int(seed_token.replace("seed", ""))
        return np.nan

    df["policy_seed"] = df.apply(lambda row: policy_seed(row, "policy_id"), axis=1)
    df["coplayer_policy_seed"] = df.apply(
        lambda row: policy_seed(row, "coplayer_policy_id"), axis=1
    )
    return df


def add_policy_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add policy type to dataframe."""

    def policy_type(row, id_key):
        _, pi_name, _ = parse_policy_id(row[id_key])
        type_tokens = [t for t in pi_name.split("_") if not t.startswith("seed")]
        return "_".join(type_tokens)

    df["policy_type"] = df.apply(lambda row: policy_type(row, "policy_id"), axis=1)
    df["coplayer_policy_type"] = df.apply(
        lambda row: policy_type(row, "coplayer_policy_id"), axis=1
    )
    return df


def import_results(
    result_file: str,
    columns_to_drop: Optional[List[str]] = None,
    clean_policy_id: bool = True,
    add_coplayer_policy_id: bool = True,
) -> pd.DataFrame:
    """Import experiment results."""
    df = pd.read_csv(result_file)

    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1, errors="ignore")

    df = add_95CI(df)
    df = add_outcome_proportions(df)

    if clean_policy_id:
        df = clean_df_policy_ids(df)

    if "policy_seed" not in df.columns:
        df = add_policy_seed(df)

    if "policy_type" not in df.columns:
        df = add_policy_type(df)

    if add_coplayer_policy_id:
        if len(df["agent_id"].unique().tolist()) == 2:
            df = add_df_coplayer_policy_id(df)
        else:
            df = add_df_multiple_coplayer_policy_id(df)

    return df


def filter_by(df: pd.DataFrame, conds: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Filter dataframe by given conditions.

    Removes any rows that do not meet the conditions.
    """
    query_strs = []
    for (k, op, v) in conds:
        if isinstance(v, str):
            q = f"({k} {op} '{v}')"
        else:
            q = f"({k} {op} {v})"
        query_strs.append(q)
    query = " & ".join(query_strs)

    filtered_df = df.query(query)
    return filtered_df


def filter_exps_by(df: pd.DataFrame, conds: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Filter experiments in dataframe by given conditions.

    Ensures all rows for an experiment where any row of that experiment
    meets the conditions are in the resulting dataframe.

    Removes any rows that do not meet the conditions and are not part of
    an experiment where at least one row (i.e. agent) meets the condition.
    """
    filtered_df = filter_by(df, conds)
    exp_ids = filtered_df["exp_id"].unique()
    df = df[df["exp_id"].isin(exp_ids)]
    return df
