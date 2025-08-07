import argparse
import json
import os
import pickle

import pandas as pd

from helper import evaluate_micro_macro


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate synthetic reports")
    parser.add_argument(
        "--responses_dir",
        type=str,
        help="Directory where response dictionary 'base_LLM.pkl' or 'fineTuned_LLM.pkl' are stored",
    )
    parser.add_argument(
        "--fine_tuned",
        type=str,
        default="False",
        help="Specify whether to evaluate the responses from 'base_LLM.pkl' or 'fineTuned_LLM.pkl'",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Path to store results.")
    parser.add_argument(
        "--target_symptoms_path",
        type=str,
        help="Path to target symptoms file (e.g. /resources/data/synthetic_reports/target_symptoms.csv).",
    )
    parser.add_argument("--ground_truth_path", type=str, help="Path to ground truth csv file")
    return parser.parse_args()


# get arguments
args = parse_args()
fine_tuned = args.fine_tuned.lower() == "true"
output_dir = args.output_dir
gt_path = args.ground_truth_path
target_symptoms_path = args.target_symptoms_path

# get target symptoms
target_symptoms = pd.read_csv(target_symptoms_path)["target_codes"].tolist()

# load ground truth
gt_loaded = pd.read_csv(gt_path, dtype={"ID": str})
gt_loaded["Codes"] = gt_loaded["Codes"].apply(json.loads)
gt_dict = dict(zip(gt_loaded["ID"], gt_loaded["Codes"]))

# load response dict
if fine_tuned:
    file_name = "fineTuned_LLM.pkl"
else:
    file_name = "base_LLM.pkl"
with open(os.path.join(args.responses_dir, file_name), "rb") as f:
    response_dict = pickle.load(f)

# process response dict
response_dict_clear = {}
for key in response_dict.keys():
    response_dict_clear[key[-3:]] = []
    for symptom in target_symptoms:
        for target in response_dict[key][symptom].keys():
            if "Yes" in response_dict[key][symptom][target]["response"]:
                response_dict_clear[key[-3:]].append(symptom)

# assemble annotation lists
sort_gt_list = []
sort_response_list = []
for key in gt_dict.keys():
    sort_gt_list.append(set(gt_dict[key]))
    sort_response_list.append(set(response_dict_clear[key]))

# calculate metrics
metrics, individual_results = evaluate_micro_macro(sort_gt_list, sort_response_list)

# Micro
print("\nEvaluation in Micro Way")
print("Micro Precision: %.4f" % metrics["micro_precision"])
print("Micro Recall: %.4f" % metrics["micro_recall"])
print("Micro F1: %.4f" % metrics["micro_f1"])

# Macro
print("\nEvaluation in Macro Way")
print("Macro Precision: %.4f" % metrics["macro_precision"])
print("Macro Recall: %.4f" % metrics["macro_recall"])
print("Macro F1: %.4f" % metrics["macro_f1"])

if output_dir:
    response_info = "base_LLM" if not fine_tuned else "fineTuned_LLM"
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(save_dir, response_info + ".csv"), index=False)
