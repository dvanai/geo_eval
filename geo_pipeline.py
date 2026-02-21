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
RUNS = int(os.environ.get("GEO_RUNS", "1"))

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


def extract_source_snippet(answer: str, url: str, window_words: int = 25) -> str:
    # Pull a small window of words around the URL to approximate the cited snippet.
    if not answer or not url:
        return ""
    tokens = answer.split()
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
                        "RunId": r.get("run_id", ""),
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
                    snippet = extract_source_snippet(chatgpt_response, url)
                    rows.append(
                        {
                            "RunId": r.get("run_id", ""),
                            "Funnel": r.get("funnel", ""),
                            "Geography": r.get("geography", ""),
                            "Category": r.get("category", ""),
                            "Prompt": r.get("original_query", ""),
                            "Full_Prompt": r.get("full_query", ""),
                            "Answer": chatgpt_response,
                            "SourceSnippet": snippet,
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
                        "RunId": r.get("run_id", ""),
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
                    snippet = extract_source_snippet(gemini_response, url)
                    rows.append(
                        {
                            "RunId": r.get("run_id", ""),
                            "Funnel": r.get("funnel", ""),
                            "Geography": r.get("geography", ""),
                            "Category": r.get("category", ""),
                            "Prompt": r.get("original_query", ""),
                            "Full_Prompt": r.get("full_query", ""),
                            "Answer": gemini_response,
                            "SourceSnippet": snippet,
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
        "RunId",
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
    "follow",
    "influence",
    "relevance",
    "subjcount",
    "subjpos",
    "uniqueness",
]

RESPONSE_KEYS = [
    "sentiment",
    "specificity",
    "brand_alignment",
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

RESPONSE_RUBRIC_TEXT = """
You are evaluating the overall response quality for Destination Vancouver. This is NOT about citations.

Rate the response on three metrics from 0 to 5 (integers only):
1) Sentiment (sentiment)
Definition: How warm, compelling, and emotionally positive is the tone when Vancouver is mentioned?
5 – Vancouver is mentioned and tone is very warm, vivid, and inviting. Vancouver is framed as inspiring, refreshing, or energizing.
4 – Vancouver is mentioned and tone is positive; Vancouver is recommended but framed more functionally or grouped with peers.
3 – Vancouver is mentioned and tone is neutral or factual; Vancouver is mentioned without emotional pull.
2 – Vancouver is mentioned, but described inaccurately, dismissively, or in a way that conflicts with brand values.
1 – Vancouver is not mentioned.

2) Specificity (specificity)
Definition: Does the response reference real, specific Vancouver places, neighbourhoods, events, or experiences?
5 – Vancouver is mentioned and there are multiple specific and accurate Vancouver references made (e.g., Stanley Park, cherry blossoms in Queen Elizabeth Park, neighbourhoods, Michelin restaurants).
4 – Vancouver is mentioned and there is at least one specific Vancouver place, experience, or neighbourhood is named.
3 – Vancouver is mentioned generally, without concrete detail.
2 – Vancouver is mentioned but information is inaccurate.
1 – Vancouver is not mentioned.

3) Brand Alignment (brand_alignment)
Definition: How well does the response reflect Destination Vancouver’s brand?
Brand pillars: Effortless, Embracing, Energizing, Fresh, Immersive Outdoors, Converging Cultures, Fresh perspectives, Wellbeing, Invigoration, Nature/Proximity of City to Nature, Culinary, Wellness, Major Events, Unique Neighbourhoods, Arts and Culture.
5 – Vancouver is mentioned and one or more of the above brand pillars are clearly reflected.
4 – Vancouver is mentioned and brand themes are touched indirectly.
3 – Vancouver is mentioned, but brand pillars are not evident.
2 – Vancouver is mentioned but themes are misaligned with our brand.
1 – Vancouver is not mentioned.

Return ONLY a valid JSON object with this exact schema and no extra commentary:
{
  "sentiment": 0,
  "specificity": 0,
  "brand_alignment": 0
}
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

def count_vancouver_mentions(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(re.findall(r"\bvancouver\b", text, flags=re.IGNORECASE))


def add_response_metrics(df: pd.DataFrame, judge_client: OpenAI) -> pd.DataFrame:
    # Add response-level metrics (brand-focused) from judge model outputs.
    df = df.copy()
    cols = {k: [] for k in RESPONSE_KEYS}
    avg_scores = []

    logging.info("Judging %s responses with %s...", len(df), JUDGE_MODEL)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            logging.info("Processed %s/%s responses", i, len(df))

        scores = judge_response_row(judge_client, row)
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


def export_response_csv(df: pd.DataFrame, path: str, dataset_name: str):
    # Write response-level evaluation CSV and summary CSV.
    df = df.copy()

    summary = (
        df.groupby(["Funnel", "Model"])
        .agg(
            Count=("Answer", "size"),
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


def _eval_key(row: dict) -> tuple:
    # Minimal unique key to avoid re-judging the same citation row.
    return (
        str(row.get("Prompt", "")).strip(),
        str(row.get("Answer", "")).strip(),
        str(row.get("Citation", "")).strip(),
        str(row.get("URL", "")).strip(),
        str(row.get("Model", "")).strip(),
        str(row.get("Funnel", "")).strip(),
        str(row.get("RunId", "")).strip(),
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
                str(r.get("RunId", "")).strip(),
            )
        )
    return keys


def run_judging():
    # Stage 3: Judge per-source rows and export evaluation CSVs.
    _, _, judge_client = get_clients()

    file_pairs = [
        (TABLE_OUTPUTS["combined"], EVAL_OUTPUTS["combined"], EVAL_PER_RUN_OUTPUTS["combined"], "Combined"),
        (TABLE_OUTPUTS["high"], EVAL_OUTPUTS["high"], EVAL_PER_RUN_OUTPUTS["high"], "High Funnel"),
        (TABLE_OUTPUTS["mid"], EVAL_OUTPUTS["mid"], EVAL_PER_RUN_OUTPUTS["mid"], "Mid Funnel"),
        (TABLE_OUTPUTS["low"], EVAL_OUTPUTS["low"], EVAL_PER_RUN_OUTPUTS["low"], "Low Funnel"),
    ]

    for input_file, output_file, per_run_output, name in file_pairs:
        if not os.path.exists(input_file):
            logging.warning("Skipping %s (not found)", input_file)
            continue

        logging.info("Processing: %s", name)
        df = pd.read_csv(input_file)
        logging.info("Loaded %s rows", len(df))

        # Skip already-judged rows to minimize API calls.
        existing_keys = _load_existing_eval_keys(per_run_output) if per_run_output else set()
        if existing_keys:
            before = len(df)
            df = df[~df.apply(lambda r: _eval_key(r), axis=1).isin(existing_keys)]
            logging.info("Skipped %s already-judged rows", before - len(df))

        if len(df) == 0:
            logging.info("No new rows to judge for %s", name)
            continue

        df = add_objective_metrics(df)
        df = add_subjective_metrics(df, judge_client)

        # Persist per-run judgments for caching.
        if per_run_output:
            if os.path.exists(per_run_output):
                existing_per_run = pd.read_csv(per_run_output)
                df = pd.concat([existing_per_run, df], ignore_index=True)
            df.to_csv(per_run_output, index=False)
            logging.info("Saved per-run judgments to %s", per_run_output)

        # Aggregate across runs and compute averaged scores.
        group_cols = [
            "Funnel",
            "Geography",
            "Category",
            "Prompt",
            "Full_Prompt",
            "Model",
            "Source",
            "Citation",
            "URL",
        ]
        agg_cols = [
            "WordCount",
            "PositionWeight",
            "PAWordCount",
            "diversity",
            "follow",
            "influence",
            "relevance",
            "subjcount",
            "subjpos",
            "uniqueness",
            "SubjectiveImpressions",
        ]

        avg_df = df.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()
        avg_df["RunCount"] = df.groupby(group_cols, dropna=False).size().values

        # Normalize objective metrics on averaged data.
        avg_df = add_objective_norms(avg_df)

        export_to_csv(avg_df, output_file, name)


def _response_key(row: dict) -> tuple:
    return (
        str(row.get("Prompt", "")).strip(),
        str(row.get("Answer", "")).strip(),
        str(row.get("Model", "")).strip(),
        str(row.get("Funnel", "")).strip(),
        str(row.get("RunId", "")).strip(),
    )


def _load_existing_response_keys(output_file: str) -> set[tuple]:
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
                str(r.get("Model", "")).strip(),
                str(r.get("Funnel", "")).strip(),
                str(r.get("RunId", "")).strip(),
            )
        )
    return keys


def build_response_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Create one row per model response (no citations).
    rows = []
    for _, r in raw_df.iterrows():
        if str(r.get("chatgpt_response", "")).strip():
            rows.append(
                {
                    "RunId": r.get("run_id", ""),
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
                    "RunId": r.get("run_id", ""),
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
        (RAW_OUTPUTS["high"], RESPONSE_EVAL_OUTPUTS["high"], RESPONSE_EVAL_PER_RUN_OUTPUTS["high"], "High Funnel"),
        (RAW_OUTPUTS["mid"], RESPONSE_EVAL_OUTPUTS["mid"], RESPONSE_EVAL_PER_RUN_OUTPUTS["mid"], "Mid Funnel"),
        (RAW_OUTPUTS["low"], RESPONSE_EVAL_OUTPUTS["low"], RESPONSE_EVAL_PER_RUN_OUTPUTS["low"], "Low Funnel"),
        (RAW_OUTPUTS["all"], RESPONSE_EVAL_OUTPUTS["combined"], RESPONSE_EVAL_PER_RUN_OUTPUTS["combined"], "Combined"),
    ]

    for input_file, output_file, per_run_output, name in file_pairs:
        if not os.path.exists(input_file):
            logging.warning("Skipping %s (not found)", input_file)
            continue

        raw_df = pd.read_csv(input_file)
        response_df = build_response_table(raw_df)
        logging.info("Loaded %s response rows for %s", len(response_df), name)

        existing_keys = _load_existing_response_keys(per_run_output) if per_run_output else set()
        if existing_keys:
            before = len(response_df)
            response_df = response_df[~response_df.apply(lambda r: _response_key(r), axis=1).isin(existing_keys)]
            logging.info("Skipped %s already-scored responses", before - len(response_df))

        if len(response_df) == 0:
            logging.info("No new responses to score for %s", name)
            continue

        response_df = add_response_metrics(response_df, judge_client)

        # Persist per-run response judgments for caching.
        if per_run_output:
            if os.path.exists(per_run_output):
                existing = pd.read_csv(per_run_output)
                response_df = pd.concat([existing, response_df], ignore_index=True)
            response_df.to_csv(per_run_output, index=False)
            logging.info("Saved per-run response scores to %s", per_run_output)

        # Aggregate across runs
        group_cols = [
            "Funnel",
            "Geography",
            "Category",
            "Prompt",
            "Full_Prompt",
            "Model",
        ]
        agg_cols = [
            "sentiment",
            "specificity",
            "brand_alignment",
            "ResponseScoreAvg",
            "VancouverMentions",
        ]
        avg_df = response_df.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()
        avg_df["RunCount"] = response_df.groupby(group_cols, dropna=False).size().values

        export_response_csv(avg_df, output_file, name)


def run_all():
    # Full pipeline in order: raw responses -> per-source tables -> judging.
    generate_raw_responses()
    build_table_views()
    run_judging()
    run_response_scoring()


if __name__ == "__main__":
    run_all()
