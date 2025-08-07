import copy
import pickle
import random
import time

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaForCausalLM

from util import HPOTree


def load_llama(llama_dir):
    """
    Loads a LLaMA causal language model with 8-bit quantization and its tokenizer from the specified directory.

    Parameters:
    -----------
    llama_dir : str
        Path to the directory containing the pretrained LLaMA model and tokenizer.

    Returns:
    --------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the LLaMA model.
    model : transformers.LlamaForCausalLM
        The loaded LLaMA causal language model with 8-bit quantization applied.

    Notes:
    ------
    - The model is loaded with automatic device mapping.
    - Quantization is configured to 8-bit for reduced memory usage.
    - The tokenizer's pad token is set to the end-of-sequence token.
    - The function prints the time taken to load the model.
    """
    start = time.time()
    config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = LlamaForCausalLM.from_pretrained(
        llama_dir,
        device_map="auto",
        quantization_config=config,
        use_cache=False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(llama_dir)
    tokenizer.pad_token = tokenizer.eos_token
    end = time.time() - start
    print("Time to load the model: ", end)
    return tokenizer, model


def complete_chat(model, tokenizer, messages, **kwargs):
    """
    Generate a chat completion response from a language model given a list of input messages.

    Parameters:
    -----------
    model : transformers.PreTrainedModel
        The language model used to generate completions.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used to process input messages and decode outputs.
    messages : list of dict
        A list of message dictionaries representing the chat history to be completed.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the model's generate method.

    Returns:
    --------
    str
        The generated text completion, decoded and with special tokens removed.

    Notes:
    ------
    - The tokenizer is expected to have an `apply_chat_template` method that formats the messages.
    - Generation uses deterministic decoding (`do_sample=False`).
    - The generated tokens corresponding to the input prompt are excluded from the output.
    """
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).to(model.device)
    num_input_tokens = len(inputs["input_ids"][0])
    generation_config = GenerationConfig(
        do_sample=False,  # No randomness
    )
    model.eval()
    with torch.no_grad():
        return tokenizer.decode(
            model.generate(**inputs, **kwargs, generation_config=generation_config)[0][num_input_tokens:],
            skip_special_tokens=True,
        )


def complete_chat_single_turn(model, tokenizer, SYSTEM_PROMPT, user: str, **kwargs):
    """
    Generate a single-turn chat completion given a system prompt and a user message.

    Parameters:
    -----------
    model : transformers.PreTrainedModel
        The language model used to generate the completion.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used to process input messages and decode outputs.
    SYSTEM_PROMPT : str
        The system-level prompt that sets the context or instructions for the model.
    user : str
        The user's input message to be responded to by the model.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the underlying `complete_chat` function.

    Returns:
    --------
    str
        The generated response from the model for the single user input, decoded and cleaned.
    """
    return complete_chat(
        model,
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )


def reformat_to_llama(user_prompts: list, assistant_responses: list, SYSTEM_PROMPT: str, tokenizer):
    """
    Prepare and format paired user prompts and assistant responses into dataloaders
    compatible with the LLaMA training framework.

    This function converts raw dialogue data into a custom dataset with tokenized inputs,
    masked labels for loss calculation, and attention masks. It registers a custom dataset
    preprocessing function, then creates and returns PyTorch dataloaders for training and evaluation.

    Parameters:
    -----------
    user_prompts : list of str
        List of user input prompts.
    assistant_responses : list of str
        List of corresponding assistant-generated responses.
    SYSTEM_PROMPT : str
        System prompt that sets the conversational context for all samples.
    tokenizer : tokenizer instance
        Tokenizer compatible with LLaMA that provides methods for applying chat templates
        and converting tokens to IDs.

    Returns:
    --------
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training dataset with tokenized and masked inputs.
    eval_dataloader : torch.utils.data.DataLoader
        DataLoader for the evaluation dataset with tokenized and masked inputs.
    """
    from copy import deepcopy
    from dataclasses import dataclass

    import datasets
    import torch
    from llama_recipes.configs import train_config as TRAIN_CONFIG
    from llama_recipes.data.concatenator import ConcatDataset
    from llama_recipes.utils.config_utils import get_dataloader_kwargs
    from llama_recipes.utils.dataset_utils import DATASET_PREPROC, get_preprocessed_dataset

    training_data = [
        {"prompt": prompt, "response": response} for prompt, response in zip(user_prompts, assistant_responses)
    ]

    # Define the dataset preprocessing function
    def get_custom_dataset(dataset_config, tokenizer, split_name):

        # Create the dataset from the custom training data
        dataset = datasets.Dataset.from_dict(
            {
                "prompt": [entry["prompt"] for entry in training_data],
                "response": [entry["response"] for entry in training_data],
            }
        )

        def apply_chat_template(sample):
            return {
                "input_ids": tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": sample["prompt"]},
                        {"role": "assistant", "content": sample["response"]},
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            }

        # Apply the chat template to each sample
        dataset = dataset.map(apply_chat_template, remove_columns=list(dataset.features))

        def create_labels_with_mask(sample):
            labels = deepcopy(sample["input_ids"])

            # Define the EOT token (end of turn token)
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            indices = [i for i, token in enumerate(sample["input_ids"]) if token == eot]
            assert len(indices) == 3, f"{len(indices)} != 3. {sample['input_ids']}"

            # Mask the loss for the system and user prompts
            labels[0:indices[1] + 1] = [-100] * (indices[1] + 1)
            assert len(labels) == len(sample["input_ids"]), f"{len(labels)} != {len(sample['input_ids'])}"

            return {"labels": labels}

        # Create labels with mask
        dataset = dataset.map(create_labels_with_mask)

        def convert_to_tensors(sample):
            return {
                "input_ids": torch.LongTensor(sample["input_ids"]),
                "labels": torch.LongTensor(sample["labels"]),
                "attention_mask": torch.tensor([1] * len(sample["labels"])),
            }

        # Convert to tensors
        dataset = dataset.map(convert_to_tensors)

        return dataset

    # Define a dataclass for your custom dataset configuration
    @dataclass
    class custom_dataset:
        dataset: str = "custom_dataset"
        train_split: str = "train"
        test_split: str = "test"
        trust_remote_code: bool = False

    # Register the dataset preprocessing function in the DATASET_PREPROC dictionary
    DATASET_PREPROC["custom_dataset"] = get_custom_dataset

    # Function to create a PyTorch dataloader
    def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
        dataset = get_preprocessed_dataset(tokenizer, dataset_config, split)
        dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)

        if split == "train" and train_config.batching_strategy == "packing":
            dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **dl_kwargs,
        )
        return dataloader

    train_config = TRAIN_CONFIG()
    train_config.context_length = 512
    train_config.batching_strategy = "packing"

    train_dataloader = get_dataloader(tokenizer, custom_dataset, train_config, "train")
    eval_dataloader = get_dataloader(tokenizer, custom_dataset, train_config, "test")

    return train_dataloader, eval_dataloader


def generate_typo1(message, nchar=1):
    """
    Typo generator based on the work of Wang et al., 2024
     @article{Wang2024,
      title = {Fine-tuning large language models for rare disease concept normalization},
      volume = {31},
      ISSN = {1527-974X},
      url = {http://dx.doi.org/10.1093/jamia/ocae133},
      DOI = {10.1093/jamia/ocae133},
      number = {9},
      journal = {Journal of the American Medical Informatics Association},
      publisher = {Oxford University Press (OUP)},
      author = {Wang,  Andy and Liu,  Cong and Yang,  Jingye and Weng,  Chunhua},
      year = {2024},
      month = jun,
      pages = {2076–2083}
    }
    """
    import random

    message = list(message)
    typo_prob = 0.2  # percent (out of 1.0) of characters to become typos

    # the number of characters that will be typos
    if nchar > 1:
        n_chars_to_flip = round(len(message) * typo_prob)
        if nchar < n_chars_to_flip:
            n_chars_to_flip = nchar  # for for example, nchar=3 but the lenght is too long
        if nchar < 1:
            nchar = 1  # at least 1 chr change
    else:
        n_chars_to_flip = nchar  # by default it is 1

    # is a letter capitalized?
    capitalization = [False] * len(message)
    # make all characters lowercase & record uppercase
    for i in range(len(message)):
        capitalization[i] = message[i].isupper()
        message[i] = message[i].lower()

    # list of characters that will be flipped
    pos_to_flip = []
    for i in range(n_chars_to_flip):
        pos_to_flip.append(random.randint(0, len(message) - 1))

    # dictionary... for each letter list of letters
    # nearby on the keyboard
    nearbykeys = {
        "a": ["q", "w", "s", "x", "z"],
        "b": ["v", "g", "h", "n"],
        "c": ["x", "d", "f", "v"],
        "d": ["s", "e", "r", "f", "c", "x"],
        "e": ["w", "s", "d", "r"],
        "f": ["d", "r", "t", "g", "v", "c"],
        "g": ["f", "t", "y", "h", "b", "v"],
        "h": ["g", "y", "u", "j", "n", "b"],
        "i": ["u", "j", "k", "o"],
        "j": ["h", "u", "i", "k", "n", "m"],
        "k": ["j", "i", "o", "l", "m"],
        "l": ["k", "o", "p"],
        "m": ["n", "j", "k", "l"],
        "n": ["b", "h", "j", "m"],
        "o": ["i", "k", "l", "p"],
        "p": ["o", "l"],
        "q": ["w", "a", "s"],
        "r": ["e", "d", "f", "t"],
        "s": ["w", "e", "d", "x", "z", "a"],
        "t": ["r", "f", "g", "y"],
        "u": ["y", "h", "j", "i"],
        "v": ["c", "f", "g", "v", "b"],
        "w": ["q", "a", "s", "e"],
        "x": ["z", "s", "d", "c"],
        "y": ["t", "g", "h", "u"],
        "z": ["a", "s", "x"],
        " ": ["c", "v", "b", "n", "m"],
    }
    # insert typos
    for pos in pos_to_flip:
        # try-except in case of special characters
        try:
            typo_arrays = nearbykeys[message[pos]]
            message[pos] = random.choice(typo_arrays)
        except:
            break

    # reinsert capitalization
    for i in range(len(message)):
        if capitalization[i]:
            message[i] = message[i].upper()

    # recombine the message into a string
    message = "".join(message)

    # show the message in the console
    return message


def generate_symptom_sentence(symptom: str) -> str:
    """
    Generate a natural language sentence describing a given symptom.

    This function selects a sentence template at random from a predefined list
    of clinical-style phrases and inserts the provided symptom into the template.
    The output simulates how symptoms might be described in medical notes or reports.

    Args:
        symptom (str): The symptom to be described in the sentence.

    Returns:
        str: A formatted sentence including the symptom.
    """
    templates = [
        "The patient presents with {symptom}.",
        "{symptom} has been reported by the patient.",
        "Clinical evaluation revealed {symptom}.",
        "The individual exhibits {symptom}.",
        "{symptom} was observed during examination.",
        "Patient complains of {symptom}.",
        "{symptom} is noted in the medical history.",
        "The patient has a history of {symptom}.",
        "Recent symptoms include {symptom}.",
        "{symptom} was detected upon assessment.",
        "{symptom} is among the patient's presenting symptoms.",
        "A diagnosis of {symptom} was considered.",
        "{symptom} has been persistent for several weeks.",
        "The patient describes experiencing {symptom}.",
        "{symptom} was confirmed through clinical evaluation.",
        "Medical records indicate {symptom}.",
        "{symptom} was reported during the consultation.",
        "The doctor noted {symptom} in the assessment.",
        "{symptom} is a significant finding in this case.",
        "The patient frequently experiences {symptom}.",
        "{symptom} is a key symptom of the current condition.",
        "Physical examination demonstrated {symptom}.",
        "The presence of {symptom} was documented.",
        "{symptom} appears to be worsening over time.",
        "{symptom} was a chief complaint during the visit.",
        "Patient describes recurrent episodes of {symptom}.",
        "The patient has been experiencing {symptom} for months.",
        "Symptoms include {symptom} among others.",
        "The onset of {symptom} was gradual.",
        "{symptom} was identified through diagnostic testing.",
        "Doctors observed {symptom} during evaluation.",
        "The medical examination highlighted {symptom}.",
        "{symptom} was first noticed a few weeks ago.",
        "{symptom} was among the initial symptoms reported.",
        "The severity of {symptom} varies over time.",
        "{symptom} is a recurrent issue for the patient.",
        "Doctors suspect an underlying cause for {symptom}.",
        "The patient recalls {symptom} starting suddenly.",
        "{symptom} has led to additional medical concerns.",
        "{symptom} was mentioned in previous consultations.",
        "{symptom} is a major concern for the patient.",
        "Medical notes frequently reference {symptom}.",
        "The patient is receiving treatment for {symptom}.",
        "{symptom} remains unresolved despite interventions.",
        "The symptoms began with {symptom} and progressed further.",
        "The doctor recorded {symptom} as part of the symptoms.",
        "Patient acknowledges {symptom} affecting daily activities.",
        "{symptom} was confirmed through imaging studies.",
        "The physician documented {symptom} during the visit.",
    ]
    return random.choice(templates).format(symptom=symptom)


def generate_negated_symptom_sentence(symptom: str) -> str:
    """
    Generate a natural language sentence expressing the absence or denial of a given symptom.

    This function randomly selects a negation template from a list of clinical-style
    phrases indicating that the patient does not have or has not experienced the specified symptom.
    The symptom is then inserted into the chosen template.

    Args:
        symptom (str): The symptom to be negated in the sentence.

    Returns:
        str: A formatted sentence indicating the absence or denial of the symptom.
    """
    templates = [
        "The patient denies any history of {symptom}.",
        "There is no evidence of {symptom}.",
        "The patient has not experienced {symptom}.",
        "No reported cases of {symptom} were noted.",
        "{symptom} has not been observed in this patient.",
        "The patient explicitly denies {symptom}.",
        "{symptom} is not present at this time.",
        "The patient does not exhibit {symptom}.",
        "No complaints related to {symptom} were reported.",
        "The patient has no history of {symptom}.",
        "There are no clinical signs of {symptom}.",
        "The examination reveals no {symptom}.",
        "No indication of {symptom} was found.",
        "The patient has never had {symptom}.",
        "No record of {symptom} in the patient's medical history.",
        "The patient denies experiencing {symptom}.",
        "{symptom} has not been a concern for the patient.",
        "The patient reports no issues with {symptom}.",
        "No clinical signs suggest {symptom}.",
        "{symptom} is absent upon examination.",
        "There is no past or current complaint of {symptom}.",
        "The patient's records do not mention {symptom}.",
        "{symptom} is not among the patient's symptoms.",
        "No history of {symptom} in previous assessments.",
        "No known episodes of {symptom}.",
        "The patient's symptoms do not include {symptom}.",
        "{symptom} is not a reported symptom.",
        "The patient has not developed {symptom}.",
        "{symptom} is not a present concern.",
        "There are no reports of {symptom}.",
        "The patient is asymptomatic for {symptom}.",
        "The current evaluation finds no {symptom}.",
        "No complaints regarding {symptom} were noted.",
        "{symptom} is not an issue for the patient.",
        "{symptom} has not been detected.",
        "The patient has no relevant symptoms of {symptom}.",
        "There is no supporting evidence of {symptom}.",
        "No signs of {symptom} were found during the check-up.",
        "{symptom} is not apparent in this case.",
        "The patient does not suffer from {symptom}.",
        "{symptom} was ruled out during the assessment.",
        "The examination confirms the absence of {symptom}.",
        "There is no indication that {symptom} is present.",
        "{symptom} does not appear in the patient's current condition.",
        "The medical team found no {symptom}.",
        "{symptom} was not noted in the assessment.",
        "{symptom} is not among the symptoms observed.",
        "The patient denies symptoms suggestive of {symptom}.",
        "The patient has remained free of {symptom}.",
    ]
    return random.choice(templates).format(symptom=symptom)


def generate_single_word(hpo_code):
    """
    Generate a list of single-word symptom names for a given HPO code, including correct and typo variants.

    For each symptom name associated with the specified HPO code, this function returns
    a list containing the original name and three generated variants with typos.

    Args:
        hpo_code (str): The Human Phenotype Ontology (HPO) code for which to generate single-word names.

    Returns:
        list of str: A list containing the correct symptom names and three typo variations for each name.
    """
    from util import HPOTree

    hpo_tree = HPOTree()
    data = hpo_tree.data

    names = data[hpo_code]["Name"]
    single_words = []
    for name in names:
        single_words.append(name)
        for i in range(3):
            single_words.append(generate_typo1(name))
    return single_words


def embed_symptom_in_prompt_v1(sents: list, hpo_code: str):
    """
    Create prompt sentences embedding detailed information about a symptom identified by an HPO code.

    For each input sentence, the prompt includes the symptom's primary label, its definition,
    and synonyms, followed by a question asking if the text segment explicitly confirms
    the presence of the symptom in a patient.

    Args:
        sents (list of str): List of text segments to be checked for the symptom.
        hpo_code (str): Human Phenotype Ontology (HPO) code identifying the symptom.

    Returns:
        list of str: A list of prompt strings, one for each input sentence, embedding symptom details.
    """
    from util import HPOTree

    hpo_tree = HPOTree()
    data = hpo_tree.data
    HPO_label = data[hpo_code]["Name"][0]  # only first element is 'Name' rest are synonyms
    Definition = data[hpo_code]["Def"][0]
    Synonyms = ", ".join(data[hpo_code]["Synonym"])

    prompt_list = []
    for sent in sents:
        prompt = (
            "The symptom "
            + HPO_label
            + " is defined as "
            + Definition
            + " "
            + HPO_label
            + " is also referred to as "
            + Synonyms
            + ". "
            + "Does the following text segment explicitly confirm that the patient has this symptom"
            + ":'"
            + sent
            + "'. "
        )
        prompt_list.append(prompt)
    return prompt_list


def convert_dict_to_lists(data_per_symptom: dict):
    """
    Convert a dictionary of symptom-related data into separate lists of prompts and responses.

    For each symptom code in the input dictionary, generates prompt sentences for both
    present and absent symptom examples using `embed_symptom_in_prompt_v1`. Corresponding
    responses are labeled as "Yes" for present symptoms and "No" for absent symptoms.

    Args:
        data_per_symptom (dict): A dictionary where keys are symptom codes (str), and values
            are dictionaries with keys "Present" and "Absent" mapping to lists of text examples
            indicating presence or absence of the symptom.

    Returns:
        tuple: Two lists -
            prompt_list (list of str): Prompts embedding symptom details and example texts.
            response_list (list of str): Corresponding "Yes" or "No" labels for symptom presence.
    """
    prompt_list = []
    response_list = []
    for code in data_per_symptom.keys():
        n_pos_examples = len(data_per_symptom[code]["Present"])
        n_neg_examples = len(data_per_symptom[code]["Absent"])
        prompt_list.extend(embed_symptom_in_prompt_v1(data_per_symptom[code]["Present"], code))
        response_list.extend(["Yes"] * n_pos_examples)
        prompt_list.extend(embed_symptom_in_prompt_v1(data_per_symptom[code]["Absent"], code))
        response_list.extend(["No"] * n_neg_examples)

    return prompt_list, response_list


def get_neg_context_dict(context_dict, codes=None, mode="hard"):
    """
    Generate a dictionary of negative context sentences for symptoms based on symptom similarity.

    This function creates negative context examples for each symptom in `context_dict` by
    selecting sentences from other symptoms' contexts. The selection strategy depends on the mode:
    - "hard": selects negative contexts from symptoms closely related to the current symptom
      (i.e., sibling terms in the HPO hierarchy excluding the symptom itself).
    - "all": selects negative contexts from all other symptoms present in `context_dict`.

    Args:
        context_dict (dict): Dictionary mapping symptom codes (str) to lists of context sentences (list of str).
        codes (list or set, optional): Subset of symptom codes to consider. If None, all symptoms are processed.
        mode (str, optional): Strategy for selecting negative contexts.
            "hard" (default) - use sibling terms in HPO hierarchy as negative context sources.
            "all" - use all other symptoms in the context_dict as negative sources.

    Returns:
        dict: A dictionary mapping each symptom code to a list of negative context sentences,
              sampled according to the selected mode.
    """
    hpo_tree = HPOTree()
    data = hpo_tree.data
    import random

    random.seed(42)

    def get_similar_terms(hpo_code):  # only neighbors (parent and child terms excluded)
        similar_terms = []
        if data[hpo_code]["Is_a"]:
            parent_term = data[hpo_code]["Is_a"][0]
            neighbor_terms = list(data[parent_term]["Son"].keys())
            neighbor_terms.remove(hpo_code)
            similar_terms.extend(neighbor_terms)
        return similar_terms

    neg_context_dict = {}
    for symptom in context_dict.keys():
        if codes is not None and symptom not in codes:  # skip if current symptom not among target symptoms
            continue
        else:
            neg_context_dict[symptom] = []
            if mode == "hard":
                terms = get_similar_terms(symptom)
            elif mode == "all":
                terms = [term for term in list(context_dict.keys()) if term != symptom]
            current_context_dict = copy.deepcopy(context_dict)

            n_pos = len(context_dict[symptom])
            current_term = 0
            n_terms = len(terms)
            for k in range(n_pos):
                try:
                    neg_sent = random.choice(current_context_dict[terms[current_term]])
                    neg_context_dict[symptom].append(neg_sent)
                    current_context_dict[terms[current_term]].remove(neg_sent)
                except Exception:
                    print("symptom: ", symptom, "has", n_terms)
                if current_term < n_terms - 1:
                    current_term += 1
                else:
                    current_term = 0

    return neg_context_dict


def filter_context_dict(context_dict: dict, hpo_codes: list):
    """
    Filter a context dictionary to retain only entries for specified HPO codes.

    Args:
        context_dict (dict): Dictionary mapping HPO codes to associated context data (e.g., sentences).
        hpo_codes (list): List of HPO codes to retain in the filtered dictionary.

    Returns:
        dict: A new dictionary containing only the entries from `context_dict` whose keys are in `hpo_codes`.
    """
    return {hpo_code: context_dict[hpo_code] for hpo_code in hpo_codes}


def get_data(context_dict_path, target_symptoms):
    """
    Load and prepare training data for a set of target symptoms, including positive and negative examples.

    This function performs the following steps:
    - Loads a context dictionary from a file mapping symptom codes to lists of example sentences.
    - Retrieves hard negative examples from sentences of neighboring symptoms.
    - Retrieves negative examples from sentences of other symptoms within the target set.
    - Generates positive and negative synthetic sentences using symptom names, synonyms, and common typos.
    - Combines all examples into a structured dictionary for each symptom.
    - Converts the structured data into flat lists of prompts and responses suitable for training.

    Args:
        context_dict_path (str): Path to a pickle file containing a dictionary mapping HPO codes
                                 to lists of example sentences.
        target_symptoms (list): List of HPO codes representing the target symptoms to prepare data for.

    Returns:
        tuple: A tuple containing two lists:
            - prompt_list (list): List of textual prompts combining symptom definitions and example sentences.
            - response_list (list): List of corresponding labels ("Yes" or "No") indicating symptom presence.

    Notes:
        The function uses helper methods to generate synthetic sentences with negation, synonyms,
        and typos to augment the dataset for better model generalization.
    """
    import random

    random.seed(42)

    hpo_codes = target_symptoms

    # load dictionary with symptom codes as keys and list of exemplary sentences as values
    with open(context_dict_path, "rb") as file:
        context_dict = pickle.load(file)

    # get context dict of hard negative examples where each symptom is assigned to a list of sentences from neighboring
    # symptoms
    neg_context_dict_filt_hard = get_neg_context_dict(context_dict, mode="hard", codes=hpo_codes)

    # get context dict of negative examples where each symptom is assigned to a list of sentences from other symptoms
    context_dict_filt = filter_context_dict(context_dict, hpo_codes)
    neg_context_dict_filt_other = get_neg_context_dict(context_dict_filt, mode="all")

    hpo_tree = HPOTree()
    data = hpo_tree.data
    data_per_symptom = {}  # dictionary of training data for each symptom
    neg_sentences = {}  # dictionary of negated sentences for each symptom
    pos_sentences = {}  # dictionary of positive sentences for each symptom
    pos_single_examples = {}  # dictionary of names, synonyms and typos for each symptom
    # dictionary of sentences belonging to other symptoms included in list of target symptoms
    neg_context_dict_filt_other_adapted = copy.deepcopy(neg_context_dict_filt_other)
    for hpo_code in hpo_codes:
        # get symptom name
        name = data[hpo_code]["Name"][0]

        # generate list containing name and variants (typos) of the symptom
        pos_single_examples[hpo_code] = generate_single_word(hpo_code)

        # limit number of negative examples from sentences belonging to other codes included in list of target symptoms
        neg_context_dict_filt_other_adapted[hpo_code] = neg_context_dict_filt_other_adapted[hpo_code][
            0 : len(pos_single_examples[hpo_code])
        ]

        # generate set of positive and negative sentences from templates using symptom name and typos
        neg_sentences[hpo_code] = []
        pos_sentences[hpo_code] = []
        typos = [generate_typo1(name) for _ in range(5)]
        for typo in typos:
            neg_sentences[hpo_code].append(generate_negated_symptom_sentence(typo))
        typos = [generate_typo1(name) for _ in range(5)]
        for typo in typos:
            pos_sentences[hpo_code].append(generate_symptom_sentence(typo))
        for i in range(10):
            neg_sentences[hpo_code].append(generate_negated_symptom_sentence(name))
            pos_sentences[hpo_code].append(generate_symptom_sentence(name))

        # generate set of positive and negative sentences from templates using symptom synonym and typos
        synonyms = data[hpo_code]["Synonym"]
        current_idx = 0
        for i in range(15):
            neg_sentences[hpo_code].append(generate_negated_symptom_sentence(synonyms[current_idx]))
            pos_sentences[hpo_code].append(generate_symptom_sentence(synonyms[current_idx]))
            if current_idx < len(synonyms) - 1:
                current_idx += 1
            else:
                current_idx = 0
        for synonym in synonyms:
            for i in range(2):
                typo_syn = generate_typo1(synonym)
                pos_sentences[hpo_code].append(generate_symptom_sentence(typo_syn))
            for i in range(2):
                typo_syn = generate_typo1(synonym)
                neg_sentences[hpo_code].append(generate_negated_symptom_sentence(typo_syn))

        # Assemble dataset for symptom
        data_per_symptom[hpo_code] = {}
        # Positive examples
        data_per_symptom[hpo_code]["Present"] = (
            context_dict[hpo_code] + pos_single_examples[hpo_code] + pos_sentences[hpo_code]
        )
        # Negative examples
        data_per_symptom[hpo_code]["Absent"] = (
            neg_context_dict_filt_hard[hpo_code]
            + neg_context_dict_filt_other_adapted[hpo_code]
            + neg_sentences[hpo_code]
        )

    # convert separated datasets into a single list of prompts and responses
    prompt_list, response_list = convert_dict_to_lists(data_per_symptom)

    return prompt_list, response_list
