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
        return "You are responding to a global traveler with no specific location bias."

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
                    "content": f"{location_context}\n\nYou are a helpful travel assistant. Provide sources where applicable.",
                },
                {"role": "user", "content": full_query},
            ],
            temperature=0.2,
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
                    "content": f"{location_context}\n\nYou are a helpful travel assistant. Provide sources where applicable.",
                },
                {"role": "user", "content": full_query},
            ],
            temperature=0.2,
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

    process_funnel("High", funnels["high"], RAW_OUTPUTS["high"], openai_client, gemini_client)
    process_funnel("Mid", funnels["mid"], RAW_OUTPUTS["mid"], openai_client, gemini_client)
    process_funnel("Low", funnels["low"], RAW_OUTPUTS["low"], openai_client, gemini_client)

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
                        "Model": "chatgpt",
                        "Source": "",
                        "Citation": "",
                        "URL": "",
                    }
                )
            else:
                for i, url in enumerate(urls, start=1):
                    rows.append(
                        {
                            "Funnel": r.get("funnel", ""),
                            "Geography": r.get("geography", ""),
                            "Category": r.get("category", ""),
                            "Prompt": r.get("original_query", ""),
                            "Full_Prompt": r.get("full_query", ""),
                            "Answer": chatgpt_response,
                            "Model": "chatgpt",
                            "Source": domain_from_url(url),
                            "Citation": f"[{i}]",
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
                        "Model": "gemini",
                        "Source": "",
                        "Citation": "",
                        "URL": "",
                    }
                )
            else:
                for i, url in enumerate(urls, start=1):
                    rows.append(
                        {
                            "Funnel": r.get("funnel", ""),
                            "Geography": r.get("geography", ""),
                            "Category": r.get("category", ""),
                            "Prompt": r.get("original_query", ""),
                            "Full_Prompt": r.get("full_query", ""),
                            "Answer": gemini_response,
                            "Model": "gemini",
                            "Source": domain_from_url(url),
                            "Citation": f"[{i}]",
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
    "follow",
    "influence",
    "relevance",
    "subjcount",
    "subjpos",
    "uniqueness",
]

RUBRIC_TEXT = """
You are evaluating how a single citation (Source [x]) contributes to a large language model answer.

Rate Source [x] on seven metrics from 0 to 5 (integers only):
1) Diversity of Ideas (diversity)
- How broad is the range of ideas or topics that Source [x] contributes to the answer?
- 0 = not cited or adds almost no new ideas.
- 5 = cited and adds a wide range of ideas, giving a comprehensive, balanced view.

2) Follow-Up Likelihood (follow)
- How likely is a user to click the citation / visit this URL for more information?
- 0 = not cited or not engaging.
- 5 = highly engaging and very likely to prompt a click.

3) Influence of Citation (influence)
- How much does the quality, completeness, and coherence of the answer depend on Source [x]?
- 0 = not cited or answer would be almost the same without it.
- 5 = central to making the answer correct, coherent, and complete.

4) Relevance of Citation to Query (relevance)
- How directly does Source [x] address the user’s query with precise, clear, useful info?
- 0 = not cited or basically irrelevant.
- 5 = highly relevant, precise, clear, and useful.

5) Subjective Count / Remembrance (subjcount)
- How much content in the answer feels like it comes from Source [x], and how memorable is it?
- 0 = not cited or barely contributes to understanding or memory.
- 5 = contributes a lot of content and is particularly memorable.

6) Subjective Position (subjpos)
- How likely is a typical user to encounter this citation while reading the answer (perceived prominence)?
- 0 = not cited or tucked away where the user is unlikely to see it.
- 5 = highly prominent; the user is almost certain to see it.

7) Uniqueness in Response (uniqueness)
- How different is the information from Source [x] compared with other sources used in the answer?
- 0 = not cited or redundant with other sources.
- 5 = clearly unique information that stands out from other sources.

Evaluation process:
1. Read the query and generated answer.
2. Focus on the parts of the answer that appear to rely on Source [x] (the citation token and its URL).
3. Consider the seven criteria above and assign each a score from 0 to 5.

Return ONLY a valid JSON object with this exact schema and no extra commentary:
{
  "diversity": 0,
  "follow": 0,
  "influence": 0,
  "relevance": 0,
  "subjcount": 0,
  "subjpos": 0,
  "uniqueness": 0
}
"""


def word_count(text: str) -> int:
    # Objective metric: basic word count.
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


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
    df["WordCount"] = df["Answer"].apply(word_count)
    df["CitationPosition"] = df["Citation"].apply(parse_citation_position)
    df["PositionWeight"] = df["CitationPosition"].apply(position_weight)
    df["PAWordCount"] = df["WordCount"] * df["PositionWeight"]

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
        return json.loads(raw)
    except Exception:
        return {k: 0 for k in SUBJECTIVE_KEYS}


def add_subjective_metrics(df: pd.DataFrame, judge_client: OpenAI) -> pd.DataFrame:
    # Add subjective metrics from judge model outputs.
    df = df.copy()
    subj_cols = {k: [] for k in SUBJECTIVE_KEYS}
    subj_avg = []

    logging.info("Judging %s rows with %s...", len(df), JUDGE_MODEL)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            logging.info("Processed %s/%s rows", i, len(df))

        scores = judge_row(judge_client, row)
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


def export_to_csv(df: pd.DataFrame, path: str, dataset_name: str):
    # Write final evaluation CSV and a summary CSV.
    df = df.copy()

    mn, mx = df["SubjectiveImpressions"].min(), df["SubjectiveImpressions"].max()
    df["SubjectiveImpressions_Norm"] = (
        (df["SubjectiveImpressions"] - mn) / (mx - mn) if mx > mn else 0.0
    )
    df["TotalMetric"] = (df["PAWordCount_Norm"] + df["SubjectiveImpressions_Norm"]) / 2

    summary = (
        df.groupby(["Funnel", "Model", "Source"])
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

    logging.info("Saved %s rows to %s (%s)", len(df), path, dataset_name)
    logging.info("Saved summary to %s", summary_path)


def _eval_key(row: dict) -> tuple:
    # Minimal unique key to avoid re-judging the same citation row.
    return (
        str(row.get("Prompt", "")).strip(),
        str(row.get("Answer", "")).strip(),
        str(row.get("Citation", "")).strip(),
        str(row.get("URL", "")).strip(),
        str(row.get("Model", "")).strip(),
        str(row.get("Funnel", "")).strip(),
    )


def _load_existing_eval_keys(output_file: str) -> set[tuple]:
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
                str(r.get("Prompt", "")).strip(),
                str(r.get("Answer", "")).strip(),
                str(r.get("Citation", "")).strip(),
                str(r.get("URL", "")).strip(),
                str(r.get("Model", "")).strip(),
                str(r.get("Funnel", "")).strip(),
            )
        )
    return keys


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

        # Skip already-judged rows to minimize API calls.
        existing_keys = _load_existing_eval_keys(output_file)
        if existing_keys:
            before = len(df)
            df = df[~df.apply(lambda r: _eval_key(r), axis=1).isin(existing_keys)]
            logging.info("Skipped %s already-judged rows", before - len(df))

        if len(df) == 0:
            logging.info("No new rows to judge for %s", name)
            continue

        df = add_objective_metrics(df)
        df = add_subjective_metrics(df, judge_client)

        # If we skipped rows earlier, append to existing output to preserve full history.
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            df = pd.concat([existing, df], ignore_index=True)

        export_to_csv(df, output_file, name)


def run_all():
    # Full pipeline in order: raw responses -> per-source tables -> judging.
    generate_raw_responses()
    build_table_views()
    run_judging()


if __name__ == "__main__":
    run_all()
