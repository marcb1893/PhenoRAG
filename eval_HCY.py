import argparse
import os
import pickle

import pandas as pd

from helper import evaluate_micro_macro
from util import HPOTree


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
        help="Path to target symptoms file (e.g. /resources/data/HCY/target_symptoms.csv).",
    )
    parser.add_argument("--ground_truth_path", type=str, help="Path to ground truth pickle file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fine_tuned = args.fine_tuned.lower() == "true"
    output_dir = args.output_dir
    gt_path = args.ground_truth_path
    target_symptoms_path = args.target_symptoms_path

    # get target symptoms
    target_symptoms = pd.read_csv(target_symptoms_path)["target_codes"].tolist()

    # load ground truth
    with open(gt_path, "rb") as f:
        gt_dict = pickle.load(f)

    # get lists accepted as valid terms for the target_symptoms
    hpo_tree = HPOTree()
    target_dict = {}
    for target_code in target_symptoms:
        child_codes = list(hpo_tree.data[target_code]["Child"].keys())
        target_dict[target_code] = [target_code] + child_codes

    # load response dict
    if fine_tuned:
        file_name = "fineTuned_LLM.pkl"  # not available for HCY
    else:
        file_name = "base_LLM.pkl"
    with open(os.path.join(args.responses_dir, file_name), "rb") as f:
        response_dict = pickle.load(f)

    # process response dict
    response_dict_clear = {}
    for key in response_dict.keys():
        response_dict_clear[key] = []
        # for symptom in response_dict[key].keys():
        for symptom in target_symptoms:
            for target in response_dict[key][symptom].keys():
                if "Yes" in response_dict[key][symptom][target]["response"]:
                    response_dict_clear[key].append(symptom)

    # assemble annotation lists
    sort_gt_list = []
    sort_response_list = []
    for key in gt_dict.keys():
        current_predictions = set(response_dict_clear[key])
        current_ground_truth = set(gt_dict[key])
        filt_predictions = []
        filt_ground_truth = []
        for symptom in target_dict.keys():
            if len(set(target_dict[symptom]) & current_predictions) > 0:
                filt_predictions.append(symptom)
            if len(set(target_dict[symptom]) & current_ground_truth) > 0:
                filt_ground_truth.append(symptom)
        sort_gt_list.append(set(filt_ground_truth))
        sort_response_list.append(set(filt_predictions))

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
