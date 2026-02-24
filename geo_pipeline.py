import os
import csv
import math
import json
import re
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd
from openai import OpenAI


# Basic, readable logging for each pipeline stage.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# =========================
# Config
# =========================

INPUT_CSV = "queries.csv"
CHATGPT_MODEL = os.environ.get("CHATGPT_MODEL", "gpt-5.2")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
# Simple pacing for judge API calls (seconds between requests).
JUDGE_SLEEP_SECONDS = float(os.environ.get("JUDGE_SLEEP_SECONDS", "0"))
RUNS = 1

RAW_OUTPUTS = {
    "high": "high_funnel_responses.csv",
    "mid": "mid_funnel_responses.csv",
    "low": "low_funnel_responses.csv",
    "all": "all_funnel_responses.csv",
}

TABLE_OUTPUTS = {
    "combined": "llm_table_view_combined_per_source.csv",
    "high": "llm_table_view_high_per_source.csv",
    "mid": "llm_table_view_mid_per_source.csv",
    "low": "llm_table_view_low_per_source.csv",
}

EVAL_OUTPUTS = {
    "combined": "geo_eval_results_combined.csv",
    "high": "geo_eval_results_high.csv",
    "mid": "geo_eval_results_mid.csv",
    "low": "geo_eval_results_low.csv",
}

EVAL_PER_RUN_OUTPUTS = {
    "combined": "geo_eval_results_combined_per_run.csv",
    "high": "geo_eval_results_high_per_run.csv",
    "mid": "geo_eval_results_mid_per_run.csv",
    "low": "geo_eval_results_low_per_run.csv",
}

RESPONSE_EVAL_OUTPUTS = {
    "combined": "geo_response_eval_combined.csv",
    "high": "geo_response_eval_high.csv",
    "mid": "geo_response_eval_mid.csv",
    "low": "geo_response_eval_low.csv",
}

RESPONSE_EVAL_PER_RUN_OUTPUTS = {
    "combined": "geo_response_eval_combined_per_run.csv",
    "high": "geo_response_eval_high_per_run.csv",
    "mid": "geo_response_eval_mid_per_run.csv",
    "low": "geo_response_eval_low_per_run.csv",
}
POSITION_LAMBDA = 0.3


# =========================
# Helpers
# =========================

def get_clients() -> tuple[OpenAI, OpenAI, OpenAI]:
    # Read API keys from environment variables.
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    if not gemini_key:
        raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.")

    openai_client = OpenAI(api_key=openai_key)
    gemini_client = OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    judge_client = OpenAI(api_key=openai_key)
    return openai_client, gemini_client, judge_client


def retry_call(fn, attempts: int = 3, base_delay: float = 1.0):
    # Lightweight retry with exponential backoff to reduce transient failures.
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            sleep_for = base_delay * (2 ** i)
            logging.warning("Call failed (%s). Retrying in %.1fs", e, sleep_for)
            time.sleep(sleep_for)


def load_queries(csv_path: str) -> dict:
    # Load input queries and derive a full prompt (query + addition).
    df = pd.read_csv(csv_path)
    logging.info("CSV columns found: %s", list(df.columns))
    logging.info("Total rows: %s", len(df))

    df["full_query"] = df["query"].astype(str) + " " + df["addition"].astype(str)

    funnels = {}
    for funnel in ["High", "Mid", "Low"]:
        funnel_df = df[df["funnel"].str.contains(funnel, case=False, na=False)]
        funnels[funnel.lower()] = funnel_df.to_dict("records")
        logging.info("%s funnel: %s queries", funnel, len(funnels[funnel.lower()]))
    return funnels


def create_location_context(geography: str) -> str:
    # Provide geographic context so responses can be localized.
    if pd.isna(geography) or str(geography).lower() == "all":
        return ""

    location_contexts = {
        "Los Angeles": "You are helping someone currently located in Los Angeles, California, USA. Consider local context, events, and preferences when relevant.",
        "San Francisco": "You are helping someone currently located in San Francisco, California, USA. Consider local context, events, and preferences when relevant.",
        "Seattle": "You are helping someone currently located in Seattle, Washington, USA. Consider local context, events, and preferences when relevant.",
        "Australia": "You are helping someone currently located in Australia. Consider local context, events, and preferences when relevant.",
    }

    return location_contexts.get(
        geography,
        f"You are helping someone currently located in {geography}. Consider local context when relevant.",
    )


def call_chatgpt(client: OpenAI, full_query: str, geography: str) -> str:
    # Call the ChatGPT model with location-aware system guidance.
    location_context = create_location_context(geography)

    def _call():
        completion = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"{location_context}\n\nProvide sources where applicable.",
                },
                {"role": "user", "content": full_query},
            ]
        )
        return completion.choices[0].message.content.strip()

    try:
        return retry_call(_call)
    except Exception as e:
        return f"ERROR: {str(e)}"


def call_gemini(client: OpenAI, full_query: str, geography: str) -> str:
    # Call the Gemini model with location-aware system guidance.
    location_context = create_location_context(geography)

    def _call():
        completion = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"{location_context}\n\nProvide sources where applicable.",
                },
                {"role": "user", "content": full_query},
            ]
        )
        return completion.choices[0].message.content.strip()

    try:
        return retry_call(_call)
    except Exception as e:
        return f"ERROR: {str(e)}"


def _raw_key(row: dict) -> tuple:
    # Minimal unique key to avoid duplicate API calls on reruns.
    return (
        str(row.get("query", "")).strip(),
        str(row.get("addition", "")).strip(),
        str(row.get("geography", "")).strip(),
        str(row.get("funnel", "")).strip(),
        str(row.get("run_id", "")).strip(),
    )


def _load_existing_raw_keys(output_file: str) -> set[tuple]:
    # Read existing raw output to skip already-processed prompts.
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
    except Exception:
        return set()
    keys = set()
    for _, r in df.iterrows():
        keys.add(
            (
                str(r.get("original_query", "")).strip(),
                str(r.get("addition", "")).strip(),
                str(r.get("geography", "")).strip(),
                str(r.get("funnel", "")).strip(),
                str(r.get("run_id", "")).strip(),
            )
        )
    return keys


def process_funnel(
    funnel_name: str,
    queries: list,
    output_file: str,
    openai_client: OpenAI,
    gemini_client: OpenAI,
):
    file_exists = os.path.exists(output_file)
    existing_keys = _load_existing_raw_keys(output_file)

    with open(output_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(output_file) == 0:
            writer.writerow(
                [
                    "timestamp",
                    "run_id",
                    "category",
                    "funnel",
                    "geography",
                    "original_query",
                    "addition",
                    "full_query",
                    "location_context",
                    "chatgpt_model",
                    "chatgpt_response",
                    "gemini_model",
                    "gemini_response",
                ]
            )

        logging.info("Processing %s funnel (%s queries)", funnel_name, len(queries))
        for i, row in enumerate(queries, 1):
            geography = row.get("geography", "All")
            location_context = create_location_context(geography)
            logging.info("[%s/%s] %s... (from: %s)", i, len(queries), row["query"][:50], geography)

            # Skip if this prompt has already been processed in this output file.
            if _raw_key(row) in existing_keys:
                logging.info("Skipping (already processed)")
                continue

            chatgpt_response = call_chatgpt(openai_client, row["full_query"], geography)
            gemini_response = call_gemini(gemini_client, row["full_query"], geography)

            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    row.get("run_id", ""),
                    row["category"],
                    row["funnel"],
                    row["geography"],
                    row["query"],
                    row["addition"],
                    row["full_query"],
                    location_context,
                    CHATGPT_MODEL,
                    chatgpt_response,
                    GEMINI_MODEL,
                    gemini_response,
                ]
            )
            logging.info("OK %s: ChatGPT & Gemini completed", geography)


def generate_raw_responses():
    # Stage 1: Generate raw responses for each funnel and combined file.
    openai_client, gemini_client, _ = get_clients()
    funnels = load_queries(INPUT_CSV)

    # Duplicate each prompt N times to capture answer variability.
    for run_id in range(1, RUNS + 1):
        for funnel_key, funnel_label in [("high", "High"), ("mid", "Mid"), ("low", "Low")]:
            for row in funnels[funnel_key]:
                row["run_id"] = run_id
            process_funnel(
                funnel_label,
                funnels[funnel_key],
                RAW_OUTPUTS[funnel_key],
                openai_client,
                gemini_client,
            )

    combined = []
    for key in ["high", "mid", "low"]:
        if os.path.exists(RAW_OUTPUTS[key]):
            combined.append(pd.read_csv(RAW_OUTPUTS[key]))
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(RAW_OUTPUTS["all"], index=False)
        logging.info("Combined raw responses saved to %s", RAW_OUTPUTS["all"])


def extract_urls(text: str) -> list[str]:
    # Extract URLs so we can compute per-citation rows for judging.
    if not isinstance(text, str):
        return []
    pattern = r"https?://[^\s)]+"
    return re.findall(pattern, text)


def domain_from_url(url: str) -> str:
    # Normalize a URL into a domain string for grouping.
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def extract_source_snippet(answer: str, citation: str = "", url: str = "", window_words: int = 25) -> str:
    # Prefer section-scoped text tied to a source line in structured answers.
    if not answer:
        return ""

    answer_text = str(answer)

    # 1) Section-based extraction: section starting at "##" or "###" that contains the URL.
    if url:
        try:
            headings = [m.start() for m in re.finditer(r"(?m)^##{2,3}\\s+.*$", answer_text)]
            if headings:
                headings.append(len(answer_text))
                for i in range(len(headings) - 1):
                    section = answer_text[headings[i] : headings[i + 1]]
                    if url in section:
                        # Cut the section at the first Source line (snippet ends before next snippet start).
                        lines = section.splitlines()
                        cut_idx = None
                        for j, line in enumerate(lines):
                            if re.search(r"(?i)^\\s*\\*?\\s*\\*\\*Source:\\*\\*.*$", line):
                                cut_idx = j
                                break
                        if cut_idx is not None:
                            section = "\n".join(lines[:cut_idx])
                        cleaned = section.strip()
                        if cleaned:
                            return cleaned
        except Exception:
            pass

    # 2) Sentence(s) containing the citation marker (e.g., [1]).
    if citation:
        sentences = re.split(r"(?<=[.!?])\\s+", answer_text.strip())
        cited = [s for s in sentences if citation in s]
        if cited:
            return " ".join(cited).strip()

    # 3) Fallback: pull a small window of words around the URL.
    if not url:
        return ""
    tokens = answer_text.split()
    idx = None
    for i, tok in enumerate(tokens):
        if url in tok:
            idx = i
            break
    if idx is None:
        return ""
    start = max(0, idx - window_words)
    end = min(len(tokens), idx + window_words + 1)
    return " ".join(tokens[start:end])


def process_funnel_data(df_funnel: pd.DataFrame, funnel_name: str | None = None) -> pd.DataFrame:
    # Convert raw responses into per-source rows for scoring.
    if funnel_name:
        logging.info("Processing %s funnel (%s rows)...", funnel_name, len(df_funnel))

    rows = []
    for _, r in df_funnel.iterrows():
        chatgpt_response = str(r.get("chatgpt_response", ""))
        gemini_response = str(r.get("gemini_response", ""))

        if not chatgpt_response.strip() and not gemini_response.strip():
            continue

        if chatgpt_response.strip():
            urls = extract_urls(chatgpt_response)
            if not urls:
                rows.append(
                    {
                        "Funnel": r.get("funnel", ""),
                        "Geography": r.get("geography", ""),
                        "Category": r.get("category", ""),
                        "Prompt": r.get("original_query", ""),
                        "Full_Prompt": r.get("full_query", ""),
                        "Answer": chatgpt_response,
                        "SourceSnippet": "",
                        "Model": "chatgpt",
                        "Source": "",
                        "Citation": "",
                        "URL": "",
                    }
                )
            else:
                for i, url in enumerate(urls, start=1):
                    citation = f"[{i}]"
                    snippet = extract_source_snippet(chatgpt_response, citation=citation, url=url)
                    rows.append(
                        {
                            "Funnel": r.get("funnel", ""),
                            "Geography": r.get("geography", ""),
                            "Category": r.get("category", ""),
                            "Prompt": r.get("original_query", ""),
                            "Full_Prompt": r.get("full_query", ""),
                            "Answer": chatgpt_response,
                            "SourceSnippet": snippet,
                            "Model": "chatgpt",
                            "Source": domain_from_url(url),
                            "Citation": citation,
                            "URL": url,
                        }
                    )

        if gemini_response.strip() and "ERROR" not in gemini_response.upper():
            urls = extract_urls(gemini_response)
            if not urls:
                rows.append(
                    {
                        "Funnel": r.get("funnel", ""),
                        "Geography": r.get("geography", ""),
                        "Category": r.get("category", ""),
                        "Prompt": r.get("original_query", ""),
                        "Full_Prompt": r.get("full_query", ""),
                        "Answer": gemini_response,
                        "SourceSnippet": "",
                        "Model": "gemini",
                        "Source": "",
                        "Citation": "",
                        "URL": "",
                    }
                )
            else:
                for i, url in enumerate(urls, start=1):
                    citation = f"[{i}]"
                    snippet = extract_source_snippet(gemini_response, citation=citation, url=url)
                    rows.append(
                        {
                            "Funnel": r.get("funnel", ""),
                            "Geography": r.get("geography", ""),
                            "Category": r.get("category", ""),
                            "Prompt": r.get("original_query", ""),
                            "Full_Prompt": r.get("full_query", ""),
                            "Answer": gemini_response,
                            "SourceSnippet": snippet,
                            "Model": "gemini",
                            "Source": domain_from_url(url),
                            "Citation": citation,
                            "URL": url,
                        }
                    )

    return pd.DataFrame(rows)


def build_table_views():
    # Stage 2: Build per-source table views for combined and per-funnel datasets.
    funnel_dfs = {}
    all_data = []

    for funnel_name in ["high", "low", "mid"]:
        file_path = RAW_OUTPUTS.get(funnel_name)
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            funnel_dfs[funnel_name] = df
            all_data.append(df)
            logging.info("Loaded %s: %s rows", funnel_name, len(df))
        else:
            logging.warning("%s not found", file_path)
            funnel_dfs[funnel_name] = pd.DataFrame()

    if not all_data:
        raise FileNotFoundError("No raw response files found.")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_out = process_funnel_data(combined_df, "Combined")

    output_columns = [
        "Funnel",
        "Geography",
        "Category",
        "Prompt",
        "Full_Prompt",
        "Model",
        "Answer",
        "SourceSnippet",
        "Source",
        "Citation",
        "URL",
    ]
    available_columns = [col for col in output_columns if col in combined_out.columns]
    combined_out = combined_out[available_columns]
    combined_out.to_csv(TABLE_OUTPUTS["combined"], index=False)
    logging.info("Saved %s rows to %s", len(combined_out), TABLE_OUTPUTS["combined"])

    for funnel_name, df_funnel in funnel_dfs.items():
        if len(df_funnel) > 0:
            out_df = process_funnel_data(df_funnel, funnel_name.capitalize())
            if len(out_df) > 0:
                available_columns = [col for col in output_columns if col in out_df.columns]
                out_df = out_df[available_columns]
                output_file = TABLE_OUTPUTS[funnel_name]
                out_df.to_csv(output_file, index=False)
                logging.info("Saved %s rows to %s", len(out_df), output_file)
            else:
                logging.warning("No valid responses found for %s", funnel_name)
        else:
            logging.warning("Skipping %s: no data", funnel_name)


# =========================
# Judging
# =========================

SUBJECTIVE_KEYS = [
    "diversity",
    "influence",
    "relevance",
    "subjcount",
    "uniqueness",
]

# Legacy subjective keys removed for current pipeline.
LEGACY_SUBJECTIVE_KEYS = []

RESPONSE_KEYS = [
    "sentiment",
    "specificity",
    "brand_alignment",
]

RUBRIC_TEXT = """
You are evaluating how a single citation (Source [x]) contributes to a large language model answer.

Rate Source [x] on seven metrics from 0 to 5 (integers only):
1) Diversity of Ideas (diversity)
Evaluation Criteria:

Diversity of Ideas (1-5) - the range of different ideas or topics discussed in the citation. This dimension expects to see how much the source contributes to a comprehensive and balanced answer to the user's query.

Evaluation Steps:

1. Read the query and generated answer carefully, noting the major points raised in the answer.
2. Read the sentences of Source [x] and assess the breadth of ideas or topics they cover and how they contribute to a comprehensive and balanced answer.
3. Assign a score for Diversity on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
4. 1 indicates that the Source [x] is not cited or does not discuss a diverse range of ideas or topics. 5 indicates that the Source [x] is cited and discusses a wide range of ideas or topics, contributing to a comprehensive and balanced answer. A number in between indicates the degree of diversity of the citation. For example, 3 would mean that Source [x] is cited, with some diversity of ideas or topics, but it is not particularly comprehensive or balanced.



2) Influence of Citation (influence)
Evaluation Criteria:

Influence of Citation (1-5) - the degree to which the answer depends on the citation. This dimension expects to see how much the source contributes to the completeness, coherence, and overall quality of the answer.

Evaluation Steps:

1. Read the query and generated answer carefully, noting the major points raised in the answer.
2. Read the sentences of Source [x] and assess how much they contribute to the completeness, coherence, and overall quality of the answer.
3. Assign a score for Influence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
4. 1 indicates that the Source [x] is not cited or does not contribute to the completeness, coherence, or quality of the answer. 5 indicates that the Source [x] is cited and contributes significantly to the completeness, coherence, and quality of the answer. A number in between indicates the degree of influence of the citation. For example, 3 would mean that Source [x] is cited, with some influence on the completeness, coherence, or quality of the answer, but it is not crucial.


3) Relevance of Citation to Query (relevance)
Evaluation Criteria:

Relevance of Citation to Query (1-5) - the degree to which the citation text is directly related to the query. This dimension expects to see how well the source addresses the user's query and provides useful and pertinent information.

Evaluation Steps:

1. Read the query and generated answer carefully, noting the major points raised in the answer.
2. Read the sentences of Source [x] and assess how directly they answer the user's query. Consider the completeness, precision, clarity, and usefulness of the information provided by Source [x].
3. Assign a score for Relevance on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
4. 1 indicates that the Source [x] is not cited or provides no relevant information. 5 indicates that the Source [x] is cited and provides highly relevant, complete, precise, clear, and useful information. A number in between indicates the degree of relevance of the information provided by Source [x]. For example, 3 would mean that Source [x] is cited, with some relevant information, but it may not be complete, precise, clear, or particularly useful.


4) Subjective Count / Remembrance (subjcount)
Evaluation Criteria:

Subjective Count/Rememberance (1-5) - the amount of content presented from the citation as perceived by the user on reading. This dimension expects to see how much the source contributes to the user's understanding and memory of the answer.

Evaluation Steps:

1. Read the query and generated answer carefully, noting the major points raised in the answer.
2. Read the sentences of Source [x] and assess how much content is presented from the citation and how much it contributes to the user's understanding and memory of the answer.
3. Assign a score for Subjective Count/Rememberance on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
4. 1 indicates that the Source [x] is not cited or does not contribute to the user's understanding or memory of the answer. 5 indicates that the Source [x] is cited and contributes significantly to the user's understanding and memory of the answer. A number in between indicates the degree of subjective count/rememberance. For example, 3 would mean that Source [x] is cited, with some contribution to the user's understanding and memory of the answer, but it is not particularly memorable.


5) Uniqueness in Response (uniqueness)
Evaluation Criteria:

Uniqueness in Response (1-5) - the unique information in answer cited to Source [x]. The dimension expects to see how much impression/visibility the source has on the user reading the generated answer. However, the impression is to be measured only because of visibility and impression.

Evaluation Steps:

1. Read the query and generated answer carefully, the major points raised in the answer.
2. Read the sentences of Source [x] and compare them to information provided by other Sources [x]. Check how unique is the information provided by Source [x] throughout the answer different from other Sources. 
3. Assign a score for Uniqueness on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
4. 1 indicates that the Source [x] is not cited. 5 indicates that the Source [x] is cited and the information is unique and different throughout the answer. A number in between indicates the degree of uniqueness of the information provided by Source [x] in the answer. For example, 3 would mean that Source [x] is cited, with some information, but is not significantly different from other Sources [x] cited in the answer. 
"""

RESPONSE_RUBRIC_TEXT = """
You are evaluating the overall response quality for Vancouver as a traveller destination. This is NOT about citations.

Rate the response on three metrics from 0 to 5 (integers only):
1) Sentiment (sentiment)
Definition: How warm, compelling, and emotionally positive is the tone when Vancouver is mentioned?
5 - Vancouver is mentioned and tone is very warm, vivid, and inviting. Vancouver is framed as inspiring, refreshing, or energizing.
4 - Vancouver is mentioned and tone is positive; Vancouver is recommended but framed more functionally or grouped with peers.
3 - Vancouver is mentioned and tone is neutral or factual; Vancouver is mentioned without emotional pull.
2 - Vancouver is mentioned, but described inaccurately, dismissively, or in a way that conflicts with brand values.
1 - Vancouver is not mentioned.

2) Specificity (specificity)
Definition: Does the response reference real, specific Vancouver places, neighbourhoods, events, or experiences?
5 - Vancouver is mentioned and there are multiple specific and accurate Vancouver references made (e.g., Stanley Park, cherry blossoms in Queen Elizabeth Park, neighbourhoods, Michelin restaurants).
4 - Vancouver is mentioned and there is at least one specific Vancouver place, experience, or neighbourhood is named.
3 - Vancouver is mentioned generally, without concrete detail.
2 - Vancouver is mentioned but information is inaccurate.
1 - Vancouver is not mentioned.

3) Brand Alignment (brand_alignment)
Definition: How well does the response reflect Destination Vancouver’s brand?
Brand pillars: Effortless, Embracing, Energizing, Fresh, Immersive Outdoors, Converging Cultures, Fresh perspectives, Wellbeing, Invigoration, Nature/Proximity of City to Nature, Culinary, Wellness, Major Events, Unique Neighbourhoods, Arts and Culture.
5 - Vancouver is mentioned and one or more of the above brand pillars are clearly reflected.
4 - Vancouver is mentioned and brand themes are touched indirectly.
3 - Vancouver is mentioned, but brand pillars are not evident.
2 - Vancouver is mentioned but themes are misaligned with our brand.
1 - Vancouver is not mentioned.
"""


def word_count(text: str) -> int:
    # Objective metric: basic word count.
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def effective_source_text(row: pd.Series) -> str:
    # Prefer source snippet when available; otherwise return empty to avoid counting full answer.
    snippet = row.get("SourceSnippet", "")
    if isinstance(snippet, str) and snippet.strip():
        return snippet
    return ""


def parse_citation_position(citation: str) -> int:
    # Objective metric: where the citation appears in the answer.
    if not isinstance(citation, str):
        return 0
    m = re.search(r"\[(\d+)\]", citation)
    return int(m.group(1)) if m else 0


def position_weight(position: int) -> float:
    # Objective metric: exponential decay by position.
    if position <= 0:
        return 0.0
    return math.exp(-POSITION_LAMBDA * (position - 1))


def add_objective_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Add objective metrics and normalize them.
    df = df.copy()
    if "SourceSnippet" in df.columns:
        df["WordCount"] = df.apply(lambda r: word_count(effective_source_text(r)), axis=1)
    else:
        df["WordCount"] = df["Answer"].apply(word_count)
    df["CitationPosition"] = df["Citation"].apply(parse_citation_position)
    df["PositionWeight"] = df["CitationPosition"].apply(position_weight)
    df["PAWordCount"] = df["WordCount"] * df["PositionWeight"]

    for col in ["WordCount", "PositionWeight", "PAWordCount"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_Norm"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0

    return df


def add_objective_norms(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize objective metrics on an aggregated dataframe.
    df = df.copy()
    for col in ["WordCount", "PositionWeight", "PAWordCount"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_Norm"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0
    return df


def build_judge_prompt(row: pd.Series) -> str:
    # Construct the judge prompt for one citation row.
    citation = str(row.get("Citation", "")).strip() or "[x]"
    return f"""
User query: "{row['Prompt']}"

Generated answer: "{row['Answer']}"

Source {citation} with URL: {row['URL']}

{RUBRIC_TEXT}
"""

def build_response_judge_prompt(row: pd.Series) -> str:
    # Construct the judge prompt for overall response scoring.
    return f"""
User query: "{row['Prompt']}"

Generated answer: "{row['Answer']}"

{RESPONSE_RUBRIC_TEXT}
"""


def judge_row(client: OpenAI, row: pd.Series) -> dict:
    # Call the judge model and parse its JSON response.
    prompt_text = build_judge_prompt(row)

    def _call():
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No explanations."},
                {"role": "user", "content": prompt_text},
            ],
        )
        return resp.choices[0].message.content.strip()

    try:
        raw = retry_call(_call)
        try:
            return json.loads(raw)
        except Exception:
            # Attempt to salvage JSON if the model added extra text.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            logging.warning("Judge returned non-JSON. Defaulting to zeros.")
            return {k: 0 for k in SUBJECTIVE_KEYS}
    except Exception:
        return {k: 0 for k in SUBJECTIVE_KEYS}

def judge_response_row(client: OpenAI, row: pd.Series) -> dict:
    # Call the judge model for overall response scoring and parse its JSON response.
    prompt_text = build_response_judge_prompt(row)

    def _call():
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No explanations."},
                {"role": "user", "content": prompt_text},
            ],
        )
        return resp.choices[0].message.content.strip()

    try:
        raw = retry_call(_call)
        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            logging.warning("Judge returned non-JSON. Defaulting to zeros.")
            return {k: 0 for k in RESPONSE_KEYS}
    except Exception:
        return {k: 0 for k in RESPONSE_KEYS}


def add_subjective_metrics(df: pd.DataFrame, judge_client: OpenAI) -> pd.DataFrame:
    # Add subjective metrics from judge model outputs.
    df = df.copy()
    subj_cols = {k: [] for k in SUBJECTIVE_KEYS}
    subj_avg = []

    dry_run = os.environ.get("JUDGE_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
    logging.info("Judging %s rows with %s...", len(df), JUDGE_MODEL)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            logging.info("Processed %s/%s rows", i, len(df))

        scores = {k: 0 for k in SUBJECTIVE_KEYS} if dry_run else judge_row(judge_client, row)
        if (not dry_run) and JUDGE_SLEEP_SECONDS > 0:
            time.sleep(JUDGE_SLEEP_SECONDS)
        vals = []
        for k in SUBJECTIVE_KEYS:
            v = scores.get(k, 0)
            if not isinstance(v, (int, float)):
                v = 0
            v = max(0, min(5, int(v)))
            subj_cols[k].append(v)
            vals.append(v)
        subj_avg.append(sum(vals) / len(SUBJECTIVE_KEYS))

    for k in SUBJECTIVE_KEYS:
        df[k] = subj_cols[k]
    df["SubjectiveImpressions"] = subj_avg
    return df

def count_vancouver_mentions(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(re.findall(r"\bvancouver\b", text, flags=re.IGNORECASE))


OTHER_CITY_PATTERNS = {
    "sydney",
    "brisbane",
    "melbourne",
    "perth",
    "san francisco",
    "sfo",
    "new york",
    "nyc",
    "los angeles",
    "la",
    "toronto",
    "montreal",
    "victoria",
    "whistler",
    "seattle",
    "chicago",
    "london",
    "paris",
    "tokyo",
    "singapore",
    "hong kong",
    "dubai",
    "mexico city",
}


def count_other_city_mentions(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    counts = 0
    for city in OTHER_CITY_PATTERNS:
        counts += len(re.findall(rf"\\b{re.escape(city)}\\b", text, flags=re.IGNORECASE))
    # Exclude Vancouver itself if present in list by mistake.
    counts -= len(re.findall(r"\\bvancouver\\b", text, flags=re.IGNORECASE))
    return max(0, counts)


def _present_cols(df: pd.DataFrame, candidates: list[str], context: str) -> list[str]:
    present = [c for c in candidates if c in df.columns]
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        logging.warning("Missing columns in %s: %s", context, ", ".join(missing))
    if not present:
        raise ValueError(f"No aggregation columns available for {context}")
    return present


def add_response_metrics(df: pd.DataFrame, judge_client: OpenAI) -> pd.DataFrame:
    # Add response-level metrics (brand-focused) from judge model outputs.
    df = df.copy()
    cols = {k: [] for k in RESPONSE_KEYS}
    avg_scores = []

    dry_run = os.environ.get("JUDGE_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
    logging.info("Judging %s responses with %s...", len(df), JUDGE_MODEL)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            logging.info("Processed %s/%s responses", i, len(df))

        scores = {k: 0 for k in RESPONSE_KEYS} if dry_run else judge_response_row(judge_client, row)
        if (not dry_run) and JUDGE_SLEEP_SECONDS > 0:
            time.sleep(JUDGE_SLEEP_SECONDS)
        vals = []
        for k in RESPONSE_KEYS:
            v = scores.get(k, 0)
            if not isinstance(v, (int, float)):
                v = 0
            v = max(0, min(5, int(v)))
            cols[k].append(v)
            vals.append(v)
        avg_scores.append(sum(vals) / len(RESPONSE_KEYS))

    for k in RESPONSE_KEYS:
        df[k] = cols[k]
    df["ResponseScoreAvg"] = avg_scores
    df["VancouverMentions"] = df["Answer"].apply(count_vancouver_mentions)
    df["OtherCityMentions"] = df["Answer"].apply(count_other_city_mentions)
    return df


def export_to_csv(df: pd.DataFrame, path: str, dataset_name: str):
    # Write final evaluation CSV and a summary CSV.
    df = df.copy()

    mn, mx = df["SubjectiveImpressions"].min(), df["SubjectiveImpressions"].max()
    df["SubjectiveImpressions_Norm"] = (
        (df["SubjectiveImpressions"] - mn) / (mx - mn) if mx > mn else 0.0
    )
    df["TotalMetric"] = (df["PAWordCount_Norm"] + df["SubjectiveImpressions_Norm"]) / 2

    summary = (
        df.groupby(["Funnel", "Geography", "Category", "Model", "Source"])
        .agg(
            Count=("Source", "size"),
            AvgTotalMetric=("TotalMetric", "mean"),
            AvgSubjective=("SubjectiveImpressions", "mean"),
            AvgPAWordCount=("PAWordCount_Norm", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary["RelevanceScore"] = summary["Count"] * summary["AvgTotalMetric"]

    df.to_csv(path, index=False)
    summary_path = os.path.splitext(path)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Extra summary: by Source only (no Model/Geography/Category).
    source_summary = (
        df.groupby(["Funnel", "Source"])
        .agg(
            Count=("Source", "size"),
            AvgTotalMetric=("TotalMetric", "mean"),
        )
        .round(3)
        .reset_index()
    )
    source_summary["RelevanceScore"] = source_summary["Count"] * source_summary["AvgTotalMetric"]
    source_summary_path = os.path.splitext(path)[0] + "_summary_source.csv"
    source_summary.to_csv(source_summary_path, index=False)

    # Extra summaries: by Geography and by Category (per Source).
    geo_summary = (
        df.groupby(["Funnel", "Geography", "Source"])
        .agg(
            Count=("Source", "size"),
            AvgTotalMetric=("TotalMetric", "mean"),
        )
        .round(3)
        .reset_index()
    )
    geo_summary["RelevanceScore"] = geo_summary["Count"] * geo_summary["AvgTotalMetric"]
    geo_summary_path = os.path.splitext(path)[0] + "_summary_geography.csv"
    geo_summary.to_csv(geo_summary_path, index=False)

    category_summary = (
        df.groupby(["Funnel", "Category", "Source"])
        .agg(
            Count=("Source", "size"),
            AvgTotalMetric=("TotalMetric", "mean"),
        )
        .round(3)
        .reset_index()
    )
    category_summary["RelevanceScore"] = category_summary["Count"] * category_summary["AvgTotalMetric"]
    category_summary_path = os.path.splitext(path)[0] + "_summary_category.csv"
    category_summary.to_csv(category_summary_path, index=False)

    # Extra summary: by URL (per Source).
    url_summary = (
        df.groupby(["Funnel", "Source", "URL"])
        .agg(
            Count=("URL", "size"),
            AvgTotalMetric=("TotalMetric", "mean"),
        )
        .round(3)
        .reset_index()
    )
    url_summary["RelevanceScore"] = url_summary["Count"] * url_summary["AvgTotalMetric"]
    url_summary_path = os.path.splitext(path)[0] + "_summary_url.csv"
    url_summary.to_csv(url_summary_path, index=False)

    logging.info("Saved %s rows to %s (%s)", len(df), path, dataset_name)
    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved source summary to %s", source_summary_path)
    logging.info("Saved geography summary to %s", geo_summary_path)
    logging.info("Saved category summary to %s", category_summary_path)
    logging.info("Saved url summary to %s", url_summary_path)


def export_response_csv(df: pd.DataFrame, path: str, dataset_name: str):
    # Write response-level evaluation CSV and summary CSV.
    df = df.copy()

    summary = (
        df.groupby(["Funnel", "Model"])
        .agg(
            Count=("ResponseScoreAvg", "size"),
            AvgResponseScore=("ResponseScoreAvg", "mean"),
            AvgVancouverMentions=("VancouverMentions", "mean"),
        )
        .round(3)
        .reset_index()
    )

    df.to_csv(path, index=False)
    summary_path = os.path.splitext(path)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)

    logging.info("Saved %s rows to %s (%s)", len(df), path, dataset_name)
    logging.info("Saved response summary to %s", summary_path)

    # Extra summaries: by Geography and by Category (per Model).
    geo_summary = (
        df.groupby(["Funnel", "Geography", "Model"])
        .agg(
            Count=("ResponseScoreAvg", "size"),
            AvgResponseScore=("ResponseScoreAvg", "mean"),
            AvgVancouverMentions=("VancouverMentions", "mean"),
            AvgOtherCityMentions=("OtherCityMentions", "mean"),
        )
        .round(3)
        .reset_index()
    )
    geo_summary_path = os.path.splitext(path)[0] + "_summary_geography.csv"
    geo_summary.to_csv(geo_summary_path, index=False)

    category_summary = (
        df.groupby(["Funnel", "Category", "Model"])
        .agg(
            Count=("ResponseScoreAvg", "size"),
            AvgResponseScore=("ResponseScoreAvg", "mean"),
            AvgVancouverMentions=("VancouverMentions", "mean"),
            AvgOtherCityMentions=("OtherCityMentions", "mean"),
        )
        .round(3)
        .reset_index()
    )
    category_summary_path = os.path.splitext(path)[0] + "_summary_category.csv"
    category_summary.to_csv(category_summary_path, index=False)

    logging.info("Saved response geography summary to %s", geo_summary_path)
    logging.info("Saved response category summary to %s", category_summary_path)

    # Combined summaries across funnels (no Funnel column).
    combined_summary = (
        df.groupby(["Model"])
        .agg(
            Count=("ResponseScoreAvg", "size"),
            AvgResponseScore=("ResponseScoreAvg", "mean"),
            AvgVancouverMentions=("VancouverMentions", "mean"),
            AvgOtherCityMentions=("OtherCityMentions", "mean"),
        )
        .round(3)
        .reset_index()
    )
    combined_summary_path = os.path.splitext(path)[0] + "_summary_all.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    combined_geo_summary = (
        df.groupby(["Geography", "Model"])
        .agg(
            Count=("ResponseScoreAvg", "size"),
            AvgResponseScore=("ResponseScoreAvg", "mean"),
            AvgVancouverMentions=("VancouverMentions", "mean"),
            AvgOtherCityMentions=("OtherCityMentions", "mean"),
        )
        .round(3)
        .reset_index()
    )
    combined_geo_summary_path = os.path.splitext(path)[0] + "_summary_all_geography.csv"
    combined_geo_summary.to_csv(combined_geo_summary_path, index=False)

    combined_category_summary = (
        df.groupby(["Category", "Model"])
        .agg(
            Count=("ResponseScoreAvg", "size"),
            AvgResponseScore=("ResponseScoreAvg", "mean"),
            AvgVancouverMentions=("VancouverMentions", "mean"),
            AvgOtherCityMentions=("OtherCityMentions", "mean"),
        )
        .round(3)
        .reset_index()
    )
    combined_category_summary_path = os.path.splitext(path)[0] + "_summary_all_category.csv"
    combined_category_summary.to_csv(combined_category_summary_path, index=False)

    logging.info("Saved response combined summary to %s", combined_summary_path)
    logging.info("Saved response combined geography summary to %s", combined_geo_summary_path)
    logging.info("Saved response combined category summary to %s", combined_category_summary_path)




def run_judging():
    # Stage 3: Judge per-source rows and export evaluation CSVs.
    _, _, judge_client = get_clients()

    file_pairs = [
        (TABLE_OUTPUTS["combined"], EVAL_OUTPUTS["combined"], "Combined"),
        (TABLE_OUTPUTS["high"], EVAL_OUTPUTS["high"], "High Funnel"),
        (TABLE_OUTPUTS["mid"], EVAL_OUTPUTS["mid"], "Mid Funnel"),
        (TABLE_OUTPUTS["low"], EVAL_OUTPUTS["low"], "Low Funnel"),
    ]

    for input_file, output_file, name in file_pairs:
        if not os.path.exists(input_file):
            logging.warning("Skipping %s (not found)", input_file)
            continue

        logging.info("Processing: %s", name)
        df = pd.read_csv(input_file)
        logging.info("Loaded %s rows", len(df))

        if len(df) == 0:
            logging.info("No new rows to judge for %s", name)
            continue

        df = add_objective_metrics(df)
        df = add_subjective_metrics(df, judge_client)

        # Backward compatibility: older cached runs may use legacy subjective keys.
        subjective_cols = [c for c in (SUBJECTIVE_KEYS + LEGACY_SUBJECTIVE_KEYS) if c in df.columns]
        if "SubjectiveImpressions" not in df.columns:
            if subjective_cols:
                df["SubjectiveImpressions"] = df[subjective_cols].mean(axis=1)
            else:
                logging.warning("No subjective columns found; defaulting SubjectiveImpressions to 0.")
                df["SubjectiveImpressions"] = 0.0

        # Aggregate across runs and compute averaged scores.
        group_cols = [
            "Funnel",
            "Geography",
            "Category",
            "Prompt",
            "Full_Prompt",
            "Answer",
            "Model",
            "Source",
            "Citation",
            "URL",
        ]
        agg_candidates = [
            "WordCount",
            "PositionWeight",
            "PAWordCount",
            "diversity",
            "influence",
            "relevance",
            "subjcount",
            "uniqueness",
            "SubjectiveImpressions",
        ]
        agg_cols = _present_cols(df, agg_candidates, f"{name} judging")

        avg_df = df.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()
        avg_df["RunCount"] = df.groupby(group_cols, dropna=False).size().values

        # Normalize objective metrics on averaged data.
        avg_df = add_objective_norms(avg_df)

        export_to_csv(avg_df, output_file, name)




def build_response_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Create one row per model response (no citations).
    rows = []
    for _, r in raw_df.iterrows():
        if str(r.get("chatgpt_response", "")).strip():
            rows.append(
                {
                    "Funnel": r.get("funnel", ""),
                    "Geography": r.get("geography", ""),
                    "Category": r.get("category", ""),
                    "Prompt": r.get("original_query", ""),
                    "Full_Prompt": r.get("full_query", ""),
                    "Model": "chatgpt",
                    "Answer": str(r.get("chatgpt_response", "")),
                }
            )
        if str(r.get("gemini_response", "")).strip() and "ERROR" not in str(r.get("gemini_response", "")).upper():
            rows.append(
                {
                    "Funnel": r.get("funnel", ""),
                    "Geography": r.get("geography", ""),
                    "Category": r.get("category", ""),
                    "Prompt": r.get("original_query", ""),
                    "Full_Prompt": r.get("full_query", ""),
                    "Model": "gemini",
                    "Answer": str(r.get("gemini_response", "")),
                }
            )
    return pd.DataFrame(rows)


def run_response_scoring():
    # Stage 4: Score overall responses for Vancouver brand criteria.
    _, _, judge_client = get_clients()

    file_pairs = [
        (RAW_OUTPUTS["high"], RESPONSE_EVAL_OUTPUTS["high"], "High Funnel"),
        (RAW_OUTPUTS["mid"], RESPONSE_EVAL_OUTPUTS["mid"], "Mid Funnel"),
        (RAW_OUTPUTS["low"], RESPONSE_EVAL_OUTPUTS["low"], "Low Funnel"),
        (RAW_OUTPUTS["all"], RESPONSE_EVAL_OUTPUTS["combined"], "Combined"),
    ]

    for input_file, output_file, name in file_pairs:
        if not os.path.exists(input_file):
            logging.warning("Skipping %s (not found)", input_file)
            continue

        raw_df = pd.read_csv(input_file)
        response_df = build_response_table(raw_df)
        logging.info("Loaded %s response rows for %s", len(response_df), name)

        if len(response_df) == 0:
            logging.info("No new responses to score for %s", name)
            continue

        response_df = add_response_metrics(response_df, judge_client)

        # Single-run output (no aggregation)
        if "Prompt" in response_df.columns:
            response_df = response_df.drop(columns=["Prompt"])
        ordered_cols = [
            "Funnel",
            "Geography",
            "Category",
            "Full_Prompt",
            "Answer",
            "Model",
            "sentiment",
            "specificity",
            "brand_alignment",
            "ResponseScoreAvg",
            "VancouverMentions",
            "OtherCityMentions",
        ]
        available_cols = [c for c in ordered_cols if c in response_df.columns]
        response_df = response_df[available_cols + [c for c in response_df.columns if c not in available_cols]]

        export_response_csv(response_df, output_file, name)


def run_all():
    # Offline-first pipeline: use existing raw CSVs and skip API calls.
    build_table_views()
    run_judging()
    run_response_scoring()


def run_offline_from_raw():
    # Offline mode: use existing raw CSVs and skip API calls for generation.
    build_table_views()
    run_judging()
    run_response_scoring()


if __name__ == "__main__":
    # Default to offline behavior in this copy.
    mode = os.environ.get("GEO_MODE", "offline").lower().strip()
    if mode == "all":
        generate_raw_responses()
    run_all()
