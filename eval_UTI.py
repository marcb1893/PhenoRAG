import argparse
import os
import pickle

import pandas as pd

from helper import evaluate_UTI


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clinical reports.")
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
        help="Path to target symptoms file (e.g. /resources/data/UTI/target_symptoms.csv).",
    )
    parser.add_argument("--ground_truth_path", type=str, help="Path to ground truth csv file")
    return parser.parse_args()


# get arguments
args = parse_args()
fine_tuned = args.fine_tuned.lower() == "true"
output_dir = args.output_dir
target_symptoms_path = args.target_symptoms_path
gt_path = args.ground_truth_path

# load ground truth data
gt_df = pd.read_csv(gt_path)
patient_data = [(str(a), str(b)) for a, b in gt_df.itertuples(index=False, name=None)]

# get target symptoms
target_symptoms = pd.read_csv(target_symptoms_path)["target_codes"].tolist()

# load response dict
if fine_tuned:
    file_name = "fineTuned_LLM.pkl"
else:
    file_name = "base_LLM.pkl"
with open(os.path.join(args.responses_dir, file_name), "rb") as f:
    response_dict = pickle.load(f)

# process response dict
confirmed = {}
for patient in response_dict.keys():
    confirmed[patient] = []
    for symptom in response_dict[patient].keys():
        for i in response_dict[patient][symptom].keys():
            if "Yes" in response_dict[patient][symptom][i]["response"]:
                confirmed[patient].append(response_dict[patient][symptom][i]["prompt"])
print("\n\n\n")
symptomatic_patients = []
for patient in confirmed.keys():
    if confirmed[patient]:
        symptomatic_patients.append(patient)
        print("Patient:", patient)
        print("Classification: CAUTI/UTI")
        print("\n Symptoms detected:")
        for item in confirmed[patient]:
            print(item)
        print("\n")
    else:
        print("Patient:", patient)
        print("Classification: AB")

# calculate performance metrics
errors, metrics = evaluate_UTI(symptomatic_patients, patient_data)
print(metrics)

# save results
if output_dir:
    save_dir = os.path.join(output_dir, file_name.split(".")[0])
    os.makedirs(save_dir, exist_ok=True)
    results_df = gt_df.copy()
    results_df["Patient_ID"] = results_df["Patient_ID"].astype(str)
    results_df["Prediction"] = results_df["Patient_ID"].map(
        lambda key: "UTI/CAUTI" if key in symptomatic_patients else "AB"
    )
    errors_marked = {}
    errors_marked["false pos"] = []
    errors_marked["false neg"] = []

    for patient in results_df["Patient_ID"]:
        if patient in errors["false pos"]:
            errors_marked["false pos"].append("X")
            errors_marked["false neg"].append("")
        elif patient in errors["false neg"]:
            errors_marked["false neg"].append("X")
            errors_marked["false pos"].append("")
        else:
            errors_marked["false pos"].append("")
            errors_marked["false neg"].append("")
    results_df = results_df.assign(**errors_marked)

    # Get the sorted keys from both dictionaries
    sorted_keys = sorted(metrics.keys())  # sorting by default in ascending order

    # Create new dictionaries with values rearranged according to sorted keys
    sorted_dict = {key: metrics[key] for key in sorted_keys}

    # Combine the sorted dictionaries into a DataFrame
    metrics_df = pd.DataFrame(list(sorted_dict.values()), index=sorted_keys)

    # save metrics
    metrics_path = os.path.join(save_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=True)

    # save individual annotations
    annotations_path = os.path.join(save_dir, "annotations.csv")
    results_df.to_csv(annotations_path, index=False)
