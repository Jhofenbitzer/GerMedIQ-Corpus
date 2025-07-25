import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration from config.yml
with open(
    "/home/hofenbitzer/datasets/germediq/scripts/github/semantic/config.yml",
    "r",
    encoding="utf-8",
) as _f:
    config = yaml.safe_load(_f)

    # Configuration values
CSV_INPUT_PATH = config["csv_input_path"]
OUTPUT_DIR_GROUP = config["output_dir_group"]
OUTPUT_DIR_CENTROID = config["output_dir_centroid"]
OUTPUT_DIR_PAIRWISE = config["output_dir_pairwise"]


combined_responses = pd.read_csv(
    CSV_INPUT_PATH,
    sep=";",
    usecols=[
        "id",
        "question",
        "response",
        "questionnaire",
        "model",
        "question_type",
        "model_domain",
    ],
)
grouped = {}

for _, row in combined_responses.iterrows():
    qid = row["id"]
    rtype = row["model"]
    rtext = row["response"]
    questionnaire = row["questionnaire"]
    question_type = row["question_type"]
    model_domain = row["model_domain"]

    # Make sure qid exists
    if qid not in grouped:
        grouped[qid] = {}

    # Make sure rtype (e.g., 'human', 'gpt-3.5', etc.) exists
    if rtype not in grouped[qid]:
        grouped[qid][rtype] = []

    grouped[qid][rtype].append((rtext, questionnaire, question_type, model_domain))

grouped_records = []
for qid, models in grouped.items():
    for model_name, responses_list in models.items():
        for response, questionnaire, question_type, model_domain in responses_list:
            grouped_records.append(
                {
                    "id": qid,
                    "model": model_name,
                    "response": response,
                    "questionnaire": questionnaire,
                    "question_type": question_type,
                    "model_domain": model_domain,
                }
            )
grouped_df = pd.DataFrame(grouped_records)
grouped_df.to_csv(OUTPUT_DIR_GROUP, index=False)
logger.info("Grouped responses saved to %s.", OUTPUT_DIR_GROUP)

# Calculate the average cosine similarity per question
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
logger.info("Loaded Sentence Transformer Model: %s.",
            model)


# Embed and compute similarities
def mean_pairwise_similarity(vecs_1, vecs_2):
    """
    Calculate pairwise similarity between different vectors.
    """
    sim_matrix = cosine_similarity(vecs_1, vecs_2)
    return sim_matrix.mean()

def mean_pairwise_similarity_single(vecs):
    """
    Calculate pairwise similarity within a set of vectors.
    """
    sim_matrix = cosine_similarity(vecs, vecs)
    # Create a boolean mask for non-diagonal entries
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)

    # Extract the off-diagonal elements
    non_diag_similarities = sim_matrix[mask]
    return non_diag_similarities.mean()

def centroid_similarity(vecs_1, vecs_2):
    """
    Calculate the similarity between two centroids.
    """
    v1_centroid = np.mean(vecs_1, axis=0)
    v2_centroid = np.mean(vecs_2, axis=0)
    return cosine_similarity([v1_centroid], [v2_centroid])[0, 0]

def centroid_similarity_rvc(vecs_1, vecs_2):
    """
    Calculate the similarity between a response and a centroid.
    """
    v1_centroid = np.mean(vecs_1, axis=0)
    v2_centroid = vecs_2
    return cosine_similarity([v1_centroid], [v2_centroid])[0, 0]


results_centroid_internal = []
results_models = []

for quid, responses in grouped.items():
    model_names = list(responses.keys())
    human_vecs_cache = {}
    logger.info("Processing Question ID: %d",
                quid)

    for i in range(len(model_names)):
        model_name_1 = model_names[i]
        texts_1 = [
            str(text) if pd.notna(text) else "<EDIT-NO-RESPONSE>"
            for text, questionnaire, question_type, model_domain in responses[
                model_name_1
            ]
        ]
        questionnaires_1 = [
            questionnaire
            for text, questionnaire, question_type, model_domain in responses[
                model_name_1
            ]
        ]
        question_types_1 = [
            question_type
            for text, questionnaire, question_type, model_domain in responses[
                model_name_1
            ]
        ]
        model_domain_1 = [
            model_domain
            for text, questionnaire, question_type, model_domain in responses[
                model_name_1
            ]
        ]
        vecs_1 = model.encode(texts_1, normalize_embeddings=True)
        if model_name_1 == "human":
            human_vecs = vecs_1
        avg_pairwise_sim = mean_pairwise_similarity_single(vecs_1)
        logger.info("Encoded %d responses for model '%s'",
                    len(texts_1),
                    model_name_1)

        centroid_similarity_records = []
        centroid_similarity_list = []
        model_centroid_human_centroid_list = []
        for j in range(len(vecs_1)):
            centroid_similarity_value = centroid_similarity_rvc(vecs_1, vecs_1[j])
            model_centroid_human_centroid = centroid_similarity_rvc(
                human_vecs, vecs_1[j]
            )
            centroid_similarity_list.append(centroid_similarity_value)
            model_centroid_human_centroid_list.append(model_centroid_human_centroid)
            # logger.info(f"Centroid distance calculated: Response {j+1}")
        avg_centroid_similarity_value = np.mean(centroid_similarity_list)
        avg_human_model_centroid_distance = np.mean(model_centroid_human_centroid_list)
        for j in range(len(vecs_1)):
            record = {
                "quid": quid,
                "model": model_name_1,
                "response": j + 1,
                "similarity_to_centroid": centroid_similarity_list[j],
                "pairwise_mean_micro": avg_pairwise_sim,
                "similarity_to_human_centroid": model_centroid_human_centroid_list[j],
                "avg_centroid_similarity": avg_centroid_similarity_value,
                "avg_human_model_centroid_distance": avg_human_model_centroid_distance,
                "questionnaire": questionnaires_1[j],
                "question_type": question_types_1[j],
                "model_domain": model_domain_1[j],
            }
            centroid_similarity_records.append(record)
        results_centroid_internal.extend(centroid_similarity_records)

        for k in range(i + 1, len(model_names)):
            model_name_2 = model_names[k]
            texts_2 = [
                str(text) if pd.notna(text) else "<EDIT-NO-RESPONSE>"
                for text, questionnaire, question_type, model_domain in responses[
                    model_name_2
                ]
            ]  # does that even make sense? I'm trying to calculate similarity values...
            questionnaires_2 = [
                questionnaire
                for text, questionnaire, question_type, model_domain in responses[
                    model_name_2
                ]
            ]
            question_types_2 = [
                question_type
                for text, questionnaire, question_type, model_domain in responses[
                    model_name_2
                ]
            ]
            model_domain_2 = [
                question_type
                for text, questionnaire, question_type, model_domain in responses[
                    model_name_2
                ]
            ]
            vecs_2 = model.encode(texts_2, normalize_embeddings=True)
            logger.info("Encoded %d responses for model '%s'",
                        len(texts_2),
                        model_name_2)

            results_models.append(
                {
                    "question_id": quid,
                    "model_1": model_name_1,
                    "model_2": model_name_2,
                    "mean_pairwise_similarity": mean_pairwise_similarity(
                        vecs_1, vecs_2
                    ),
                    "mean_centroid_comparison": centroid_similarity(vecs_1, vecs_2),
                    "questionnaire": questionnaires_1[0] if questionnaires_1 else None,
                    "question_type": question_types_1[0] if question_types_1 else None,
                    "model_domain": model_domain_1[0] if model_domain_1 else None,
                }
            )
            logger.info("Compared model '%s' with model '%s'",
            model_name_1,
            model_name_2)

results_centroid_internal_df = pd.DataFrame(results_centroid_internal)
results_centroid_internal_df = results_centroid_internal_df[
    [
        "quid",
        "model",
        "response",
        "similarity_to_centroid",
        "pairwise_mean_micro",
        "similarity_to_human_centroid",
        "avg_centroid_similarity",
        "avg_human_model_centroid_distance",
        "questionnaire",
        "question_type",
        "model_domain",
    ]
]
results_centroid_internal_df.to_csv(OUTPUT_DIR_CENTROID, index=False)
logger.info("Saved to similarity_results_centroid_internal.csv")

results_models_df = pd.DataFrame(results_models)
results_models_df = results_models_df[
    [
        "question_id",
        "model_1",
        "model_2",
        "mean_pairwise_similarity",
        "mean_centroid_comparison",
        "questionnaire",
        "question_type",
        "model_domain",
    ]
]
results_models_df.to_csv(OUTPUT_DIR_PAIRWISE, index=False)
logger.info("Saved to similarity_results_models.csv")
