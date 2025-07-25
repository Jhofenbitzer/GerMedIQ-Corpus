######################## LLM-based Data Augmentation in a Zero-Shot Manner ########################
###################################################################################################
######################################## Part of the Paper ########################################
####################### GerMedIQ: A Resource for Simulated and Synthesized ########################
#######################       Anamnesis Interview Responses in German      ########################
######################################### (ACL SRW, 2025) #########################################
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
    Gemma3ForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
)
import yaml

# Load configuration from config.yml
with open(
    "config.yml",
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
PADDING = config.get("padding", "max_length")

# Define the prompt template using dedent to clean up the formatting
PROMPT_TEMPLATE = dedent(
    """
    Du erhältst gleich eine Interviewfrage aus einem standardisierten medizinischen 
    Anamnesefragebogen auf deutsch. Bitte beantworte die Frage auf deutsch, so als 
    wenn du ein realer Patient in der Routineversorgung wärst.
    Frage: {question}
    Antwort:
    """
).strip()


class StopOnStop(StoppingCriteria):
    """
    Define the Stopping Criteria for the extraction of generated responses.
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


def zero_shot_response_generation(question: str, model, tokenizer) -> str:
    """
    Generate a zero-shot response for a given question using the provided model and tokenizer.
    """
    max_input_length = MAX_INPUT_LENGTH
    max_new_tokens = MAX_NEW_TOKENS
    min_new_tokens = MIN_NEW_TOKENS

    # Set pad token to eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        PROMPT_TEMPLATE.format(question=question),
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding=PADDING,
    ).to(model.device)

    prompt_tok_len = inputs.input_ids.shape[1]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["!", "?"]]

    stopping_criteria = StoppingCriteriaList(
        [StopOnStop(stop_token_ids, prompt_tok_len)]
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,  # disable default EOS stopping
            bad_words_ids=[[tokenizer.eos_token_id]],  # forbid EOS as a sampled token
            stopping_criteria=stopping_criteria,
        )

    new_ids = outputs[0, prompt_tok_len:].tolist()
    if len(new_ids) == 0:
        new_ids = outputs[0]
    # decode without dropping EOS, then manually strip it
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    # remove all your “<end_of_turn>” markers and collapse excess newlines
    clean = (
        raw.replace("<end_of_turn>", "")  # drop the marker
        .replace("\r", "")  # if any CRs
        .replace("\n\n", "\n")  # collapse double-linebreaks
        .replace("<pad>", "")
        .replace("<unk>", "")
        .replace("<|start_header_id|>", "")
        .replace("<|end_header_id|>", "")
        .replace("assistant", "")
        .strip()  # trim leading/trailing whitespace
    )
    answer = clean.replace(tokenizer.eos_token, "").strip()

    return answer


def gpu_usage():
    """
    Determine which GPUs have a headroom of 20 % and define the BNB Config.
    """
    # Set the target GPU device (ensure the device exists)
    gpus = GPUtil.getGPUs()
    max_memory = {
        gpu.id: f"{int(gpu.memoryFree * 0.8)}MiB"  # reserve ~20% headroom
        for gpu in gpus
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # or "fp4"
        bnb_4bit_compute_dtype=torch.float16,
    )
    return max_memory, bnb_config


def get_model(model_name, bnb_config, max_memory):
    """
    Load the correct model and tokenizer.
    """
    if model_name in ("google/flan-t5-base", "CLARA-MeD/flan-t5-base"):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",  # pipeline parallelism across ALL GPUs
            max_memory=max_memory,  # your custom upper bounds per card
            low_cpu_mem_usage=True,
        )
    elif model_name in (
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float32)
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",  # pipeline parallelism across ALL GPUs
            max_memory=max_memory,  # your custom upper bounds per card
            low_cpu_mem_usage=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",  # pipeline parallelism across ALL GPUs
            max_memory=max_memory,  # your custom upper bounds per card
            low_cpu_mem_usage=True,
        )  # defers weights to CPU until needed)

    model.eval()  # Set to evaluation mode

    return model, tokenizer


def write_csv_after_run(model_alias, run, generated_responses_dict, model_name):
    """
    Write the CSV files after each run.
    """
    # Define a unique CSV filename that includes the run number
    output_csv = os.path.join(
        OUTPUT_DIR, f"generated_responses_{model_alias}_run{run}.csv"
    )
    df = pd.DataFrame(
        list(generated_responses_dict.items()), columns=["question", "response"]
    )
    df["id"] = df["question"].factorize()[0] + 1
    # Map questionnaire to each question without exploding rows
    orig = pd.read_csv(
        "/home/hofenbitzer/datasets/germediq/germediq.csv", delimiter=";"
    )[["question", "questionnaire", "question_type"]]
    questionnaire_map = orig.drop_duplicates("question").set_index("question")[
        "questionnaire"
    ]
    question_type_map = orig.drop_duplicates("question").set_index("question")[
        "question_type"
    ]
    # apply both maps
    df["questionnaire"] = df["question"].map(questionnaire_map)
    df["question_type"] = df["question"].map(question_type_map)

    # now you can safely subset
    df = df[["id", "question", "response", "questionnaire", "question_type"]]
    df.to_csv(output_csv, index=False)
    logger.info(
        "Saved responses for model %s (Run %d) to %s", model_name, run, output_csv
    )

    return output_csv


def process_model(model_name: str, model_alias: str, questions: set, run: int) -> str:
    """
    Load the model and its tokenizer, generate responses for all questions,
    save results to a CSV file with a unique name (including the run number),
    and clear GPU memory afterward.

    Returns:
        The path to the CSV file generated in this run.
    """
    max_memory, bnb_config = gpu_usage()
    logger.info("Loading model: %s (Run %d)", model_name, run)

    model, tokenizer = get_model(model_name, bnb_config, max_memory)

    generated_responses_dict = {}
    ids = []
    total_questions = len(questions)
    # Sorting questions for consistent ordering
    for idx, question in enumerate(questions, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total_questions:
            logger.info(
                "Processing question %d/%d for model %s (Run %d)",
                idx,
                total_questions,
                model_alias,
                run,
            )
        response = zero_shot_response_generation(question, model, tokenizer)
        if len(response) == 0:
            response = "<EDIT-NO-RESPONSE>"
        generated_responses_dict[question] = response
        ids.append(idx)

    output_csv = write_csv_after_run(
        model_alias, run, generated_responses_dict, model_name
    )

    # Clear GPU memory
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    return output_csv


def main():
    """
    Main Body of the Script: Execute the llm_augmentation.py script.
    """
    # Load the dataset
    csv_file = CSV_INPUT_PATH
    df = pd.read_csv(csv_file, delimiter=";")
    questions = df["question"].unique().tolist()
    logger.info(
        "Loaded dataset from %s with %d unique questions.", csv_file, len(questions)
    )

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

    # Process each model for 39 runs
    for model_alias, model_name in models.items():
        for run in range(1, 6):
            output_csv = process_model(model_name, model_alias, questions, run)
            csv_info.append({"model": model_alias, "run": run, "csv_path": output_csv})

    # Combine all CSV files into one DataFrame with additional columns: Model and Run
    combined_df = pd.DataFrame()
    for info in csv_info:
        df_run = pd.read_csv(info["csv_path"])
        # Create an index column to preserve original order
        df_run["orig_index"] = range(len(df_run))
        df_run["model"] = info["model"]
        df_run["run"] = info["run"]
        combined_df = pd.concat([combined_df, df_run], ignore_index=True)

    # If needed, sort by Model, Run, and original index to restore the exact order per CSV
    combined_df.sort_values(by=["id", "model", "run"], inplace=True)

    # Optionally drop the helper column
    combined_df.drop(columns=["orig_index"], inplace=True)

    combined_output_csv = os.path.join(OUTPUT_DIR, "generated_responses_combined.csv")
    combined_df.to_csv(combined_output_csv, index=False)
    logger.info("Combined CSV saved to %s", combined_output_csv)


if __name__ == "__main__":
    main()
