import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Locate the final-answer delimiter using flexible variants.
FINAL_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:###\s*)?(?:final\s*[_\-\s]*answer|final\s*[_\-\s]*result|answer)\s*[:\-–]?\s*",
    re.IGNORECASE,
)
# Numbered and bullet step markers to segment reasoning.
STEP_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:step\s*)?(?:\d+)[\.:)]\s*", re.IGNORECASE
)
BULLET_MARKER_PATTERN = re.compile(r"(?:^|\n)\s*[-*•]\s+")
# Patterns for robust numeric extraction from model answers.
LATEX_FRAC_RE = re.compile(
    r"(?P<sign>-?)\\frac\{\s*(?P<num>-?\d+)\s*\}\{\s*(?P<den>-?\d+)\s*\}"
)
SIMPLE_FRAC_RE = re.compile(r"(?<!\d)(?P<sign>-?)\s*(?P<num>-?\d+)\s*/\s*(?P<den>-?\d+)(?!\d)")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def split_reasoning_and_final(output_text: str) -> Tuple[str, str]:
    """Split model output into reasoning part and trailing final answer section."""
    if not isinstance(output_text, str):
        return "", ""
    matches = list(FINAL_MARKER_PATTERN.finditer(output_text))
    if not matches:
        return output_text.strip(), ""
    match = matches[-1]
    reasoning = output_text[: match.start()].strip()
    final_section = output_text[match.end() :].strip()
    return reasoning, final_section


def clean_step_text(text: str) -> str:
    """Trim whitespace but keep any numbering/bullets."""
    return text.strip()


def parse_steps(reasoning_text: str) -> Tuple[List[str], str]:
    """Extract ordered steps plus any preamble prefix before the first step."""
    if not reasoning_text:
        return [], ""
    text = reasoning_text.strip()
    matches = list(STEP_MARKER_PATTERN.finditer(text))
    steps: List[str] = []
    prefix = ""
    if matches:
        prefix_raw = text[: matches[0].start()].strip()
        prefix = prefix_raw
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            step = text[start:end].lstrip("\n").strip()
            if step:
                steps.append(step)
    else:
        lines = text.splitlines()
        bullet_lines = []
        first_bullet_idx = None
        for idx, line in enumerate(lines):
            if BULLET_MARKER_PATTERN.match(f"\n{line}") and clean_step_text(line):
                bullet_lines.append(clean_step_text(line))
                if first_bullet_idx is None:
                    first_bullet_idx = idx
        if first_bullet_idx is not None:
            prefix = "\n".join(lines[:first_bullet_idx]).strip()
        if len(bullet_lines) > 1:
            steps = bullet_lines
        else:
            paragraph_split = [clean_step_text(p) for p in re.split(r"\n\s*\n", text)]
            paragraph_split = [p for p in paragraph_split if p]
            if len(paragraph_split) > 1:
                steps = paragraph_split
                prefix = ""
            else:
                line_split = [
                    clean_step_text(line)
                    for line in lines
                    if line.strip()
                ]
                if len(line_split) > 1:
                    steps = line_split
                    prefix = ""
                else:
                    steps = [clean_step_text(text)]
                    prefix = ""
    return steps, prefix


def extract_final_answer(final_section: str) -> Tuple[str, Optional[float]]:
    """Return first-line raw final answer text and a numeric parse (if any)."""
    if not isinstance(final_section, str):
        return "", None
    text = final_section.strip()
    if not text:
        return "", None

    first_line = text.splitlines()[0].strip()

    def _try_parse_latex_fraction(s: str) -> Optional[float]:
        m = LATEX_FRAC_RE.search(s)
        if not m:
            return None
        sign = -1.0 if m.group("sign") == "-" else 1.0
        num = float(m.group("num"))
        den = float(m.group("den"))
        if den == 0:
            return None
        return sign * (num / den)

    def _try_parse_simple_fraction(s: str) -> Optional[float]:
        m = SIMPLE_FRAC_RE.search(s)
        if not m:
            return None
        sign = -1.0 if m.group("sign") == "-" else 1.0
        num = float(m.group("num"))
        den = float(m.group("den"))
        if den == 0:
            return None
        return sign * (num / den)

    def _sum_all_numbers(s: str) -> Optional[float]:
        nums = NUMBER_RE.findall(s)
        if not nums:
            return None
        total = 0.0
        for n in nums:
            try:
                total += float(n)
            except Exception:
                continue
        return total

    def extract_pred_value(s: str) -> Optional[float]:
        if not isinstance(s, str):
            return None
        s = s.strip()
        if not s:
            return None

        boxed = BOXED_RE.search(s)
        if boxed:
            inner = boxed.group(1)
            for fn in (_try_parse_latex_fraction, _try_parse_simple_fraction):
                val = fn(inner)
                if val is not None:
                    return val
            val = _sum_all_numbers(inner.replace(",", ""))
            if val is not None:
                return val

        for fn in (_try_parse_latex_fraction, _try_parse_simple_fraction):
            val = fn(s)
            if val is not None:
                return val

        val = _sum_all_numbers(s.replace(",", ""))
        if val is not None:
            return val
        return None

    numeric_value = extract_pred_value(text)
    if numeric_value is not None and np.isnan(numeric_value):
        numeric_value = None
    return first_line, numeric_value


def build_reasoning_json(prompt: str, prefix: str, steps: List[str]) -> str:
    """Build JSON list of cumulative states (prompt + all steps so far)."""
    traces: List[str] = []
    prompt_block = prompt.strip()
    prefix_block = prefix.strip()
    for idx in range(len(steps)):
        body_parts = []
        if prefix_block:
            body_parts.append(prefix_block)
        body_parts.append("\n\n".join(steps[: idx + 1]))
        combined_body = "\n\n".join(body_parts)
        parts = [combined_body]
        if prompt_block:
            parts.insert(0, prompt_block)
        traces.append("\n\n".join(parts))
    return json.dumps(traces, ensure_ascii=True)


def compute_label(expected_result: float, final_numeric: Optional[float]) -> int:
    """Label 1 if truncated predicted answer mismatches expected result."""
    if final_numeric is None or pd.isna(expected_result):
        return 1
    truncated_final = math.trunc(final_numeric)
    if math.isclose(truncated_final, expected_result, rel_tol=1e-9, abs_tol=1e-6):
        return 0
    return 1


def process_file(input_path: str, output_path: str, drop_missing_steps: bool) -> None:
    df = pd.read_csv(input_path)
    reasoning_json_values = []
    labels = []
    parsed_flags = []
    for _, row in df.iterrows():
        prompt = row.get("prompt__qwen-3b-instruct", "")
        output_text = row.get("output__qwen-3b-instruct", "")
        reasoning_text, final_section = split_reasoning_and_final(output_text)
        steps, prefix = parse_steps(reasoning_text)
        parsed_flags.append(bool(steps))
        reasoning_json_values.append(build_reasoning_json(prompt, prefix, steps))
        _, numeric_final = extract_final_answer(final_section)
        expected = pd.to_numeric(row.get("result"), errors="coerce")
        labels.append(compute_label(expected, numeric_final))
    df["reasoning_steps_json"] = reasoning_json_values
    df["steps_parsed"] = parsed_flags
    df["label"] = labels
    if drop_missing_steps:
        df = df[df["steps_parsed"]]
    df.to_csv(output_path, index=False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "qwen-3b-instruct-gsm8k-responses.csv"
    default_output = script_dir / "qwen-3b-instruct-gsm8k-responses-processed.csv"
    parser = argparse.ArgumentParser(
        description="Process Qwen GSM8K responses to extract reasoning traces and correctness labels."
    )
    parser.add_argument(
        "--input",
        default=str(default_input),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Path to write the processed CSV with reasoning JSON and labels.",
    )
    parser.add_argument(
        "--drop-missing-steps",
        action="store_true",
        help="Drop rows where reasoning steps could not be parsed.",
    )
    args = parser.parse_args()
    process_file(args.input, args.output, drop_missing_steps=args.drop_missing_steps)


if __name__ == "__main__":
    main()
