"""Temporal graph-feature LightGBM ensemble for the BT4012 competition."""

from pathlib import Path
import os
import warnings

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR = Path(os.environ.get(
    "BT4012_DATA_DIR",
    r"C:\Users\regan\.cache\kagglehub\competitions\bt-4012-competition-2026",
))
OUT = Path("submission_improved_temporal_graph_ensemble.csv")


GRAPH_COLUMNS = [
    "graph_pagerank", "component_id", "component_size",
    "clustering_coeff", "k_core", "reciprocity", "log_in_degree",
    "log_out_degree", "in_degree", "out_degree", "total_degree",
    "flow_ratio", "log_component_size",
]


def add_historical_graph_features(train, test, edges):
    all_rows = pd.concat(
        [train[["txId", "time_step"]], test[["txId", "time_step"]]],
        ignore_index=True,
    ).drop_duplicates("txId")
    tx_time = all_rows.set_index("txId")["time_step"]
    source, target = edges.columns[:2]
    timed = edges.copy()
    timed["source_time"] = timed[source].map(tx_time)
    timed["target_time"] = timed[target].map(tx_time)
    timed = timed.dropna(subset=["source_time", "target_time"])

    train = train.copy()
    test = test.copy()
    for frame in (train, test):
        for column in GRAPH_COLUMNS:
            frame[column] = 0.0

    for step in sorted(pd.concat([train["time_step"], test["time_step"]]).unique()):
        historical = timed[
            (timed["source_time"] < step) & (timed["target_time"] < step)
        ]
        if historical.empty:
            continue
        graph = nx.from_pandas_edgelist(
            historical, source=source, target=target, create_using=nx.DiGraph
        )
        undirected = graph.to_undirected()
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
        clustering = nx.clustering(undirected)
        core = nx.core_number(undirected) if undirected.number_of_edges() else {}
        component_id, component_size = {}, {}
        components = sorted(nx.connected_components(undirected), key=lambda c: min(c))
        for number, component in enumerate(components):
            for node in component:
                component_id[node] = number
                component_size[node] = len(component)

        values = {}
        for node in graph.nodes:
            outgoing = set(graph.successors(node))
            incoming = set(graph.predecessors(node))
            neighbours = outgoing | incoming
            in_degree = len(incoming)
            out_degree = len(outgoing)
            size = component_size.get(node, 0)
            values[node] = [
                pagerank.get(node, 0.0), component_id.get(node, -1),
                size, clustering.get(node, 0.0), core.get(node, 0),
                len(outgoing & incoming) / len(neighbours) if neighbours else 0.0,
                np.log1p(in_degree), np.log1p(out_degree), in_degree,
                out_degree, in_degree + out_degree,
                (in_degree + 1.0) / (out_degree + 1.0), np.log1p(size),
            ]
        for frame in (train, test):
            mask = frame["time_step"].eq(step)
            if mask.any():
                frame.loc[mask, GRAPH_COLUMNS] = (
                    frame.loc[mask, ["txId"]]["txId"]
                    .map(values)
                    .apply(lambda row: row if isinstance(row, list) else [0.0] * len(GRAPH_COLUMNS))
                    .tolist()
                )
    return train, test


def make_model(pos_weight, seed, leaves=31):
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=600, learning_rate=0.03,
        num_leaves=leaves, min_child_samples=30, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_weight,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    edges = pd.read_csv(DATA_DIR / "txs_edgelist.csv")
    train["label"] = pd.to_numeric(train["label"], errors="coerce")
    train, test = add_historical_graph_features(train, test, edges)

    labelled = train["label"].isin([0, 1])
    features = [
        c for c in train.columns
        if c not in {"txId", "label", "index", "component_id"}
        and c in test.columns
    ]
    graph_features = [c for c in GRAPH_COLUMNS if c in features and c != "component_id"]
    tabular_features = [c for c in features if c not in graph_features]
    sets = {
        "all": features,
        "tabular": tabular_features,
        "graph": graph_features,
    }
    y = train.loc[labelled, "label"].astype("int8")
    x = train.loc[labelled]
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    labelled_steps = np.sort(x["time_step"].unique())
    fractions = [(0.50, 0.65), (0.65, 0.80), (0.80, 1.00)]
    oof = pd.DataFrame(np.nan, index=x.index, columns=list(sets))
    for fold, (start_fraction, end_fraction) in enumerate(fractions):
        start = labelled_steps[int(len(labelled_steps) * start_fraction)]
        end = labelled_steps[min(int(len(labelled_steps) * end_fraction), len(labelled_steps) - 1)]
        fit = x["time_step"] < start
        valid = (x["time_step"] >= start) & (x["time_step"] <= end)
        for number, (name, columns) in enumerate(sets.items()):
            model = make_model(pos_weight, 1000 + fold * 10 + number, 15 if name == "graph" else 31)
            model.fit(x.loc[fit, columns], y.loc[fit])
            oof.loc[valid, name] = model.predict_proba(x.loc[valid, columns])[:, 1]
            print(f"fold={fold + 1} {name} AUC={roc_auc_score(y.loc[valid], oof.loc[valid, name]):.6f}")

    valid = oof.notna().all(axis=1)
    blend_y, blend_oof = y.loc[valid], oof.loc[valid]
    best_score, best_weights = -1.0, None
    for all_weight in np.arange(0, 0.81, 0.1):
        for tab_weight in np.arange(0, 1.01 - all_weight, 0.1):
            weights = (all_weight, tab_weight, 1 - all_weight - tab_weight)
            prediction = sum(weight * blend_oof[name] for weight, name in zip(weights, sets))
            score = roc_auc_score(blend_y, prediction)
            if score > best_score:
                best_score, best_weights = score, weights
    print(f"OOF AUC={best_score:.6f}; weights={best_weights}")

    # Add only high-confidence pseudo-labels from rows before the final
    # labelled period. This never uses test rows or validation labels.
    pseudo_cutoff = train.loc[labelled, "time_step"].quantile(0.75)
    pseudo_source = train.index[
        (~labelled) & (train["time_step"] <= pseudo_cutoff)
    ]
    base_fit = labelled & (train["time_step"] <= pseudo_cutoff)
    base = make_model(pos_weight, 3000)
    base.fit(train.loc[base_fit, features], train.loc[base_fit, "label"])
    pseudo_prob = base.predict_proba(train.loc[pseudo_source, features])[:, 1]
    keep = np.maximum(pseudo_prob, 1 - pseudo_prob) >= 0.98
    pseudo_index = pseudo_source[keep]
    x_final = pd.concat([train.loc[labelled], train.loc[pseudo_index]])
    y_final = pd.concat([
        train.loc[labelled, "label"].astype("int8"),
        pd.Series((pseudo_prob[keep] >= 0.5).astype("int8"), index=pseudo_index),
    ])
    print(f"pseudo-labels added={len(pseudo_index):,}")

    test_prediction = pd.DataFrame(index=test.index)
    for number, (name, columns) in enumerate(sets.items()):
        model = make_model(pos_weight, 4000 + number, 15 if name == "graph" else 31)
        model.fit(x_final[columns], y_final)
        test_prediction[name] = model.predict_proba(test[columns])[:, 1]
    pred = sum(weight * test_prediction[name] for weight, name in zip(best_weights, sets))

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sample["target"] = np.clip(pred, 0, 1)
    sample.to_csv(OUT, index=False)
    print(f"saved {OUT.resolve()} shape={sample.shape}")


if __name__ == "__main__":
    main()
