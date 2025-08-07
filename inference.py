import os
import pickle
import time
from dataclasses import asdict

import hydra
import torch
from llama_recipes.configs import lora_config as LORA_CONFIG
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer

from helper import (
    SymptomScoreCalculator,
    encode_dict,
    getReports,
    load_stanza,
    load_txt,
    prepare_context,
    segment_dict,
    split_sents,
)
from llama_helper import complete_chat_single_turn
from llama_helper import embed_symptom_in_prompt_v1 as embed_symptom_in_prompt
from llama_helper import load_llama


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_symptom_scan(cfg: DictConfig):

    # Load llama base model and tokenizer
    tokenizer, model = load_llama(cfg.llama_dir_base)

    # Set system prompt
    SYSTEM_PROMPT = (
        "You are a medical expert deciding whether a patient has a certain symptom. Answer only with 'Yes' or 'No'."
    )

    if cfg.llama_dir_ft:
        lora_config = LORA_CONFIG()
        lora_config.r = 8
        lora_config.lora_alpha = 32

        peft_config = LoraConfig(**asdict(lora_config))

        model = get_peft_model(model, peft_config)

        # Load the state_dict from the fine-tuned model
        model_path = cfg.llama_dir_ft
        state_dict = torch.load(model_path, map_location="cpu")

        # Load the weights into the model
        model.load_state_dict(state_dict)

    # Move the model to evaluation mode
    model.eval()

    # Get symptoms to check for from context_data
    target_symptoms = [
        code.replace("_", ":").split(".")[0] for code in os.listdir(cfg.context_dir) if code.endswith(".txt")
    ]

    # Load sentence transformer
    sent_model = SentenceTransformer(cfg.sent_transformer_dir)

    # Initialize stanza pipeline with default parameters
    nlp_tokenized = load_stanza(stanza_dir=cfg.stanza_dir, mode="TOKENIZER")

    # Context data
    context_dict = load_txt(cfg.context_dir)
    prep_context_dict = prepare_context(context_dict)
    enc_prep_context_dict = encode_dict(prep_context_dict, sent_model)

    # Load patient data
    if cfg.input_dir == "UTI":
        report_dict, summary_dict = getReports()
    else:
        summary_dict = load_txt(cfg.input_dir)

    summary_sent_dict = segment_dict(summary_dict, nlp_tokenized)  # segment text into sentences
    summary_sent_dict_splitted = split_sents(summary_sent_dict)
    enc_summary_sent_dict = encode_dict(summary_sent_dict_splitted, sent_model)  # encode patient data

    # Initialize retriever
    symptom_calculator = SymptomScoreCalculator(enc_prep_context_dict)

    # Retrieve sentences
    filtered_sents = symptom_calculator.symptom_sents(
        enc_summary_sent_dict, summary_sent_dict_splitted, target_symptoms
    )

    # Generate prompts
    for key in filtered_sents.keys():
        for symptom in target_symptoms:
            filtered_sents[key][symptom]["prompts"] = embed_symptom_in_prompt(
                filtered_sents[key][symptom]["top_sents"], symptom
            )

    # Get responses
    response_dict = {}
    for patient in filtered_sents.keys():
        response_dict[patient] = {}
        for symptom in target_symptoms:
            response_dict[patient][symptom] = {}
            prompts = filtered_sents[patient][symptom]["prompts"]
            for i, prompt in enumerate(prompts):
                response_dict[patient][symptom][i] = {}
                response_dict[patient][symptom][i]["prompt"] = prompt
                print("Patient:", patient)
                print("Symptom:", filtered_sents[patient][symptom]["symptom name"])
                print("Prompt:", prompt)
                response = complete_chat_single_turn(model, tokenizer, SYSTEM_PROMPT, prompt, max_new_tokens=128)
                print("Response:", response)
                response_dict[patient][symptom][i]["response"] = response
                time.sleep(2)

    # Save results
    response_info = "base_LLM" if not cfg.llama_dir_ft else "fineTuned_LLM"
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, response_info + ".pkl"), "wb") as f:
        pickle.dump(response_dict, f)


if __name__ == "__main__":
    run_symptom_scan()
