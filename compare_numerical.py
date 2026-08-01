import io
import math
import argparse
import pandas as pd

import logging

logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--compare_type', type=str, required=True, choices=['numerical', 'csv'])
    parser.add_argument('--tolerance', type=float, default=0.01, help='Relative tolerance for numeric equality (default 0.01 = 1%).')
    return parser.parse_args()



def _to_float(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        s = str(value).strip()
    except Exception:
        return None
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def is_within_tolerance(ground_truth, candidate, tolerance):
    ground_truth = _to_float(ground_truth)
    candidate = _to_float(candidate)
    if candidate is None:
        return False
    if ground_truth == 0:
        return abs(candidate) <= tolerance
    return abs(candidate - ground_truth) / abs(ground_truth) <= tolerance


def _parse_csv_cell(text):
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return pd.read_csv(io.StringIO(s), dtype=str, keep_default_na=False)
    except Exception as e:
        logger.warning(f'Failed to parse CSV cell: {type(e).__name__}: {e}')
        return None


INDEX_COLS = ('year', 'corporate', 'date')


def compare_csv_cell(ground_truth, candidate, tolerance):
    ground_truth = _parse_csv_cell(ground_truth)
    if ground_truth is None or ground_truth.empty:
        return float('nan')
    candidate = _parse_csv_cell(candidate)

    index_cols = [c for c in INDEX_COLS if c in ground_truth.columns]
    if index_cols:
        ground_truth = ground_truth.set_index(index_cols)

    if candidate is not None and index_cols:
        for c in index_cols:
            if c not in candidate.columns:
                candidate[c] = pd.NA
        candidate = candidate.set_index(index_cols)

    total = 0
    correct = 0
    for col in ground_truth.columns:
        for idx, gt_val in ground_truth[col].items():
            gt_f = _to_float(gt_val)
            if gt_f is None:
                continue
            total += 1
            if col not in candidate.columns or idx not in candidate.index:
                continue
            cand_val = candidate.at[idx, col]
            if isinstance(cand_val, pd.Series):
                cand_val = cand_val.iloc[0]
            if is_within_tolerance(gt_f, cand_val, tolerance):
                correct += 1
    if total == 0:
        return float('nan')
    return correct / total


def eval_numerical(df, tolerance):
    missing = [c for c in ('answer', 'response_answer') if c not in df.columns]
    if missing:
        raise ValueError(f'Input CSV is missing required column(s): {missing}')
    df[f'accuracy_answer_{tolerance}'] = [
        1.0 if is_within_tolerance(gt, pred, tolerance) else 0.0
        for gt, pred in zip(df['answer'], df['response_answer'])
    ]


def eval_csv(df, tolerance):
    # for col in ('data', 'data_indicator'):
    for col in ('data_indicator',):
        response_col = f'response_{col}'
        missing = [c for c in (col, response_col) if c not in df.columns]
        if missing:
            raise ValueError(f'Input CSV is missing required column(s): {missing}')
        df[f'accuracy_{col}_{tolerance}'] = [
            compare_csv_cell(gt, pred, tolerance)
            for gt, pred in zip(df[col], df[response_col])
        ]



def main(output_path, input_path, compare_type, tolerance):
    df = pd.read_csv(input_path)

    if compare_type == 'numerical':
        eval_numerical(df, tolerance)
    else:
        eval_csv(df, tolerance)

    df.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')



if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
