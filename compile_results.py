import json
import numpy as np
import pandas as pd


METHODS = {
    "LSTM-PPO": "reports/smiles_lstm_ppo.json",
    "LSTM-HC": "reports/smiles_lstm_hc.json",
    "SMILES GA": "reports/smiles_ga.json",
    "Graph GA": "reports/graph_ga.json",
    "Best of dataset": "reports/best_from_chembl.json",
}

metric_order = [
    "Score",
    "Target Response",
    "Blood-Brain Barrier",
    # "QED Score",
    "CNS MPO",
    "Synthetic Accessibility",
]

benchmark_order = [
    "Alzheimer MPO",
    "Schizophrenia MPO",
    # "Lung cancer MPO"
    "Parkinson MPO",
]

report_results = list()


def get_metadata_keys(metadata):
    keys = list(metadata.keys())

    target = [k for k in keys if "GeometricMeanScoringFunction" in k]
    target_scores = metadata[target[0]] if len(target) > 0 else None
        
    # qed = [k for k in keys if "RdkitScoringFunction" in k]
    # qed_scores = metadata[qed[0]] if len(qed) > 0 else None

    bbb = [k for k in keys if "TargetResponseScoringFunction" in k]
    bbb_scores = metadata[bbb[0]] if len(bbb) > 0 else None

    sa = [k for k in keys if "SyntheticAccessibilityScoringFunction" in k]
    sa_scores = metadata[sa[0]] if len(sa) > 0 else None

    cns = [k for k in keys if "CNS_MPO_ScoringFunction" in k]
    cns_scores = metadata[cns[0]] if len(cns) > 0 else None

    return {
        "Target Response": target_scores,
        "Blood-Brain Barrier": bbb_scores,
        # "QED Score": qed_scores,
        "Synthetic Accessibility": sa_scores,
        "CNS MPO": cns_scores        
    }


for method in METHODS:
    report = METHODS[method]

    with open(report) as f:
        results = json.load(f)["results"]
        
        for result in results:
            results_info = dict()

            results_info["Method"] = method
            results_info["Benchmark"] = result["benchmark_name"]
            results_info["Score"] = f"{result['score']:.5f}"

            metadata = result["metadata"]
            scores_dict = get_metadata_keys(metadata)

            for score_key in scores_dict:
                scores = scores_dict[score_key]
                if scores is None:
                    continue

                mean_score = np.mean(scores)
                std_score = np.std(scores)

                results_info[score_key] = f"{mean_score:.3f} ± {std_score:.3f}"
            
            report_results.append(results_info.copy())

df = pd.DataFrame(report_results)   

# Reorder the benchmarks
df["Benchmark"] = pd.Categorical(df["Benchmark"], benchmark_order, ordered=True)

dfs = list()

for metric in metric_order:
    df_score = df.pivot_table(
        index=["Benchmark"],
        columns=["Method"],
        values=[metric],
        aggfunc=lambda x: x,
    )

    df_score.columns = df_score.columns.droplevel(0)
    df_score["Metric"] = metric

    dfs.append(df_score)

concat_dfs = pd.concat(dfs, axis=0).sort_values(by=["Benchmark"])

concat_dfs["Metric"] = pd.Categorical(
    concat_dfs["Metric"], categories=metric_order, ordered=True
)

concat_dfs[['Metric'] + list(METHODS.keys())].to_csv('reports/MTDD-Benchmarks.csv')