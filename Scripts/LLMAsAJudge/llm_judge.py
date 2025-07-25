###################################### LLM-as-a-judge Script ######################################
###################################################################################################
######################################## Part of the Paper ########################################
####################### GerMedIQ: A Resource for Simulated and Synthesized ########################
#######################       Anamnesis Interview Responses in German      ########################
#######################                   (ACL SRW, 2025)                  ########################
###################################################################################################
###################### (C) Justin Hofenbitzer, Technical University of Munich #####################
###################################################################################################


import gc
import logging
import os
from textwrap import dedent
import GPUtil
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
    LogitsProcessorList,
    LogitsProcessor,
    Gemma3ForCausalLM,
)
import yaml

# Load configuration from config.yml
with open(
    "/home/hofenbitzer/datasets/germediq/scripts/github/judge/config.yml",
    "r",
    encoding="utf-8",
) as _f:
    config = yaml.safe_load(_f)

# Configure environment for CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.get(
    "pytorch_cuda_alloc_conf", "expandable_segments:True"
)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration values
CSV_INPUT_PATH = config["csv_input_path"]
OUTPUT_DIR = config["output_dir"]
MAX_INPUT_LENGTH = config.get("max_input_length", 300)
MAX_NEW_TOKENS = config.get("max_new_tokens", 50)
MIN_NEW_TOKENS = config.get("min_new_tokens", 1)

# Define the prompt template using dedent to clean up the formatting
PROMPT_TEMPLATE = dedent(
    """
    You are an expert in medical interviews and your task is to evaluate the quality of a given 
    response to a medical questionnaire question, both written in German. Your rating should 
    consider the appropriateness of a response. A response is considered appropriate if it 
    answers the question properly, it is natural, coherent and contextually suitable. Rate each 
    response on a scale from 1 (not appropriate) to 5 (very appropriate). Please, respond only 
    with a number and do not justify your rating.
    Question: {question}
    Answer: {answer}
    Judgement:
    """
).strip()


class RemoveNaNInf(LogitsProcessor):
    """
    A logits processor to clamp NaN/Inf logits to large finite values
    so sampling will not produce invalid probabilities.
    """

    def __call__(self, input_ids, scores):
        # Replace NaN/Inf in logits with large finite negatives (for nan/infs) 
        # or positives (if needed)
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


class StopOnStop(StoppingCriteria):
    """
    Define the stopping criteria for the LLM-judgments.
    """
    def __init__(self, stop_token_ids, prompt_length):
        self.stop_token_ids = set(stop_token_ids)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if the last token is part of the following list
        if input_ids.shape[1] <= self.prompt_length:
            return False
        last_token_id = input_ids[0, -1].item()
        return last_token_id in self.stop_token_ids


def judgement_generation(
    model,
    tokenizer,
    prompts,
    stop_tokens,
    max_input_length,
    max_new_tokens,
    min_new_tokens,
):
    """
    Generate judgements for a batch of prompts using the given model and tokenizer.
    Returns a dict mapping batch index to the cleaned judgement string.
    """
    # Tokenize all prompts at once
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding="longest",
    ).to(next(model.parameters()).device)

    # Set up stopping criteria
    prompt_len = inputs.input_ids.shape[1]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
    stopping = StoppingCriteriaList([StopOnStop(stop_token_ids, prompt_len)])

    # Generate model outputs
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
            bad_words_ids=[[tokenizer.eos_token_id]],
            stopping_criteria=stopping,
            logits_processor=LogitsProcessorList([RemoveNaNInf()]),
        )

    # Decode and clean each output
    generated = {}
    for i, seq in enumerate(outputs):
        new_ids = seq.tolist()[prompt_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        clean = (
            raw.replace("<end_of_turn>", "")
            .replace("\r", "")
            .replace("\n\n", "\n")
            .replace("<pad>", "")
            .replace("<unk>", "")
            .strip()
        )
        judgment = clean.replace(tokenizer.eos_token, "").strip()
        generated[i] = judgment
    return generated


def process_model(
    name: str, model: str, questions_list: list, responses_list: list
) -> str:
    """
    Load the model and its tokenizer, generate a judgment for all responses,
    save results to a CSV file with a unique name,
    and clear GPU memory afterward.

    Returns:
        The path to the CSV file generated.
    """

    # Classify each row by model domain: human, biomedical, or general
    def classify_domain(model_name):
        """
        Helper function to classify the model domain.
        """
        if model_name == "human":
            return "human"
        if "medical" in model_name.lower() or "bio" in model_name.lower():
            return "biomedical"
        return "general"

    # Get the GPU's free memory and reserve 20 % of headroom
    gpus = GPUtil.getGPUs()
    max_memory = {gpu.id: f"{int(gpu.memoryFree * 0.8)}MiB" for gpu in gpus}

    # Configure Bits & Bytes; Apply quantization to reduce memory footprints
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the corresponding model and its tokenizer
    logger.info("Loading model: %s.", name)

    # For the flanT5 models, we use the model-specific tokenizer and the corresponding model class
    if model in ("google/flan-t5-base", "CLARA-MeD/flan-t5-base"):
        tokenizer = T5Tokenizer.from_pretrained(model)
        tokenizer.padding_side = "left"
        m = T5ForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.float16,  # Convert model weights to half precision
            trust_remote_code=True,
            quantization_config=bnb_config,  # Apply the quantization
            device_map="auto",  # Automatically map the computation to free GPU devices
            max_memory=max_memory,  # Only assign devices with 20 % headroom
            low_cpu_mem_usage=True,
        )  # Optimize for RAM

    # For the Gemma 3 models, we use the 'AutoTokenizer' and the corresponding model class
    elif name in ("google/gemma-3-4b-it"):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.padding_side = "left"
        m = Gemma3ForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        # Ensure model is in half precision
        m = m.half()

    # All other models are loaded with the 'AutoTokenizer' and the 'AutoModelForCausalLM' class
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
    # Set the model to evaluation mode
    m.eval()

    # Prepare batched prompts
    prompts = [
        PROMPT_TEMPLATE.format(question=q, answer=r)
        for q, r in zip(questions_list, responses_list)
    ]
    stop_tokens = ["!", "?"]
    # Generate all judgements in one shot
    generated_dict = judgement_generation(
        model=m,
        tokenizer=tokenizer,
        prompts=prompts,
        stop_tokens=stop_tokens,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
    )
    # Reconstruct mapping to include question, response, and judgement
    generated_judgements_dict = {
        idx: [questions_list[idx], responses_list[idx], generated_dict[idx]]
        for idx in generated_dict
    }

    # Define a unique CSV filename and create dataframe from the obtained judgments
    output_csv = f"{OUTPUT_DIR}_{name}.csv"
    orig = pd.read_csv(".../GerMedIQ-Corpus/GerMedIQ-LLM.csv", delimiter=";")[
        [
            "question",
            "response",
            "questionnaire",
            "question_type",
            "model",
            "model_domain",
        ]
    ]
    df = pd.DataFrame.from_dict(
        generated_judgements_dict,
        orient="index",
        columns=["question", "response", "judgements"],
    )
    df["judge"] = name
    df["judge_domain"] = classify_domain(name)
    df["id"] = df["question"].factorize()[0] + 1

    # Map questionnaire to each question without exploding rows
    questionnaire_map = orig.drop_duplicates("question").set_index("question")[
        "questionnaire"
    ]
    question_type_map = orig.drop_duplicates("question").set_index("question")[
        "question_type"
    ]
    df["questionnaire"] = df["question"].map(questionnaire_map)
    df["question_type"] = df["question"].map(question_type_map)

    # Merge model and domain based on question + response and safe the final CSV
    model_meta = orig.drop_duplicates(subset=["question", "response"])[
        ["question", "response", "model", "model_domain"]
    ]
    df = df.merge(model_meta, on=["question", "response"], how="left")
    df = df[
        [
            "id",
            "question",
            "response",
            "questionnaire",
            "question_type",
            "model",
            "model_domain",
            "judge",
            "judge_domain",
            "judgements",
        ]
    ]
    df.to_csv(output_csv, index=False)
    logger.info("Saved responses for model %s to %s",
    name,
    output_csv)

    # Clear GPU memory
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    return output_csv


def main():
    """
    Execute the main part of the LLM-as-a-judge script.
    """
    ## Load the LLM responses and separate questions from responses
    path = CSV_INPUT_PATH
    df = pd.read_csv(path, sep=";")
    questions_list = list(df["question"])
    responses_list = list(df["response"])
    logger.info("I read the file from %s and turned it into two lists.", path)

    # Mapping of model alias to their Hugging Face identifiers
    models = {
        "gemma3-4B-standard": "google/gemma-3-4b-it",
        "flanT5-base-medical": "CLARA-MeD/flan-t5-base",
        "Mistral-124B-large-standard": "mistralai/Mistral-Large-Instruct-2411",
        "R1-Qwen-8B-standard": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "ministral-8B-standard": "mistralai/Ministral-8B-Instruct-2410",
        "llama-3.3-70B-standard": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen2.5-7B-standard": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7B-medical": "prithivMLmods/Qwen-UMLS-7B-Instruct",
        "flanT5-base-standard": "google/flan-t5-base",
        "mistral-7B-medical": "BioMistral/BioMistral-7B",
        "mistral-7B-standard": "mistralai/Mistral-7B-Instruct-v0.1",
        "biogpt-medtext-347M-medical": "AventIQ-AI/BioGPT-MedText",
        "biogpt-medical": "microsoft/biogpt",
        "phi-4-mini-standard": "microsoft/Phi-4-mini-instruct",
        "bloom-6B4-german-standard": "malteos/bloom-6b4-clp-german",
        "llama-3.2-1B-standard": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-1B-medical": "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025",
        "llama-3.2-3B-standard": "meta-llama/Llama-3.2-3B-Instruct",
    }

    # List to hold CSV file information for later combination
    csv_info = []

    # Iterate over every LLM and obtain the judgements
    for name, model in models.items():
        output_csv = process_model(name, model, questions_list, responses_list)
        csv_info.append({"csv_path": output_csv})

    # Combine all CSV files into one DataFrame
    combined_df = pd.DataFrame()
    for info in csv_info:
        single_df = pd.read_csv(info["csv_path"])
        combined_df = pd.concat([combined_df, single_df], ignore_index=True)

    # Sort the dataframe by by ID and write the final dataframe to CSV
    combined_df.sort_values(by=["id"], inplace=True)
    combined_output_csv = OUTPUT_DIR
    combined_df.to_csv(combined_output_csv, index=False)
    logger.info("Combined CSV saved to %s.", combined_output_csv)


# Execute the script
if __name__ == "__main__":
    main()
