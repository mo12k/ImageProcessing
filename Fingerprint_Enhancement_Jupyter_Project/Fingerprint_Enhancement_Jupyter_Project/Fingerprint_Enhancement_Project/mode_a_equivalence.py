from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


NUMERIC_RTOL = 3e-4
NUMERIC_ATOL = 5e-4
MAX_MISMATCH_EXAMPLES = 5

CSV_FILES = [
    "dataset_inventory.csv",
    "real_subject_counts.csv",
    "dataset_split_manifest.csv",
    "development_baseline_metrics.csv",
    "gabor_screening_parameter_metrics.csv",
    "gabor_screening_parameter_summary.csv",
    "gabor_diagnostic_metrics.csv",
    "member1_parameter_search.csv",
    "member1_parameter_metrics.csv",
    "member2_parameter_search.csv",
    "member2_parameter_metrics.csv",
    "member3_parameter_search.csv",
    "member3_parameter_metrics.csv",
    "member4_parameter_search.csv",
    "member4_parameter_metrics.csv",
    "development_member_metrics.csv",
    "development_member_comparison.csv",
    "validation_baseline_metrics.csv",
    "validation_member_metrics.csv",
    "validation_member_comparison.csv",
    "altered_structural_metrics.csv",
    "altered_structural_summary.csv",
]

JSON_FILES = [
    "dataset_split_audit.json",
    "final_selected_configuration.json",
    "validation_summary.json",
    "environment_report.json",
]

ENVIRONMENT_PROVENANCE_KEYS = {
    "matplotlib",
    "numpy",
    "opencv",
    "pandas",
    "pillow",
    "platform",
    "python",
    "scikit_image",
    "scipy",
}


def drop_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    runtime_columns = [
        column
        for column in df.columns
        if "runtime" in column.lower() or column.lower() == "seconds"
    ]
    return df.drop(columns=runtime_columns, errors="ignore")


def remove_runtime_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_runtime_keys(item)
            for key, item in value.items()
            if "runtime" not in key.lower() and key.lower() != "seconds"
        }
    if isinstance(value, list):
        return [remove_runtime_keys(item) for item in value]
    return value


def run_equivalence_check(
    script_output_dir: Path,
    notebook_output_dir: Path,
    csv_files: Iterable[str] = CSV_FILES,
    json_files: Iterable[str] = JSON_FILES,
) -> dict[str, Any]:
    script_output_dir = Path(script_output_dir)
    notebook_output_dir = Path(notebook_output_dir)

    records = [
        compare_csv_file(script_output_dir, notebook_output_dir, name)
        for name in csv_files
    ]
    records.extend(
        compare_json_file(script_output_dir, notebook_output_dir, name)
        for name in json_files
    )
    records.append(compare_figures(script_output_dir, notebook_output_dir))

    maximum_numeric_difference = _maximum_numeric_difference(records)
    final_selection = _final_selection_summary(script_output_dir, notebook_output_dir)
    summary = _build_summary(records, final_selection)

    return {
        "all_passed": all(record["status"] == "PASS" for record in records),
        "tolerances": {
            "rtol": NUMERIC_RTOL,
            "atol": NUMERIC_ATOL,
            "rule": (
                "numeric values must satisfy abs(script - notebook) <= "
                "atol + rtol * scale; delta_* columns use the paired "
                "before/after metric columns as scale when present"
            ),
        },
        "ignored_fields": ["runtime columns", "runtime keys", "seconds"],
        "summary": summary,
        "final_selection": final_selection,
        "maximum_numeric_difference": maximum_numeric_difference,
        "records": records,
    }


def compare_csv_file(
    script_output_dir: Path,
    notebook_output_dir: Path,
    name: str,
) -> dict[str, Any]:
    script_path = script_output_dir / name
    notebook_path = notebook_output_dir / name
    base_record = {"file": name, "kind": "csv"}
    if not script_path.exists() or not notebook_path.exists():
        return {
            **base_record,
            "status": "FAIL",
            "failure_type": "missing_file",
            "detail": (
                f"missing expected file; script_exists={script_path.exists()}, "
                f"notebook_exists={notebook_path.exists()}"
            ),
        }

    left = drop_runtime_columns(pd.read_csv(script_path))
    right = drop_runtime_columns(pd.read_csv(notebook_path))
    if list(left.columns) != list(right.columns):
        return {
            **base_record,
            "status": "FAIL",
            "failure_type": "columns",
            "detail": "column structure mismatch",
            "script_columns": list(left.columns),
            "notebook_columns": list(right.columns),
        }
    if len(left) != len(right):
        return {
            **base_record,
            "status": "FAIL",
            "failure_type": "row_count",
            "detail": f"row-count mismatch; script={len(left)}, notebook={len(right)}",
            "script_rows": len(left),
            "notebook_rows": len(right),
        }

    numeric_mismatches: list[dict[str, Any]] = []
    text_mismatches: list[dict[str, Any]] = []
    maximum_numeric_difference: dict[str, Any] | None = None

    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            column_mismatches, column_max = _compare_numeric_column(left, right, column)
            numeric_mismatches.extend(column_mismatches)
            if _is_larger_numeric_difference(column_max, maximum_numeric_difference):
                maximum_numeric_difference = column_max
        else:
            mismatch = _compare_exact_column(left, right, column)
            if mismatch is not None:
                text_mismatches.append(mismatch)

    if text_mismatches:
        return {
            **base_record,
            "status": "FAIL",
            "failure_type": "categorical",
            "detail": f"categorical/text mismatch in {text_mismatches[0]['column']}",
            "mismatches": text_mismatches[:MAX_MISMATCH_EXAMPLES],
            "max_numeric_difference": maximum_numeric_difference,
        }
    if numeric_mismatches:
        first = numeric_mismatches[0]
        return {
            **base_record,
            "status": "FAIL",
            "failure_type": "numeric",
            "detail": (
                f"numeric mismatch in {first['column']} at row {first['row']}; "
                f"abs_diff={first['absolute_difference']:.12g}, "
                f"allowed={first['allowed_tolerance']:.12g}"
            ),
            "mismatches": numeric_mismatches[:MAX_MISMATCH_EXAMPLES],
            "max_numeric_difference": maximum_numeric_difference,
        }

    detail = "matched exactly after runtime columns were ignored"
    if maximum_numeric_difference is not None:
        detail = (
            "matched within numeric tolerance after runtime columns were ignored; "
            f"max_abs_diff={maximum_numeric_difference['absolute_difference']:.12g}"
        )
    return {
        **base_record,
        "status": "PASS",
        "detail": detail,
        "rows": len(left),
        "columns": list(left.columns),
        "max_numeric_difference": maximum_numeric_difference,
    }


def compare_json_file(
    script_output_dir: Path,
    notebook_output_dir: Path,
    name: str,
) -> dict[str, Any]:
    script_path = script_output_dir / name
    notebook_path = notebook_output_dir / name
    base_record = {"file": name, "kind": "json"}
    if not script_path.exists() or not notebook_path.exists():
        return {
            **base_record,
            "status": "FAIL",
            "failure_type": "missing_file",
            "detail": (
                f"missing expected file; script_exists={script_path.exists()}, "
                f"notebook_exists={notebook_path.exists()}"
            ),
        }

    left = remove_runtime_keys(json.loads(script_path.read_text(encoding="utf-8")))
    right = remove_runtime_keys(json.loads(notebook_path.read_text(encoding="utf-8")))

    if name == "environment_report.json":
        return _compare_environment_report(name, left, right)
    if name in {"dataset_split_audit.json", "final_selected_configuration.json"}:
        return _compare_exact_json(name, left, right)
    return _compare_tolerant_json(name, left, right)


def compare_figures(script_output_dir: Path, notebook_output_dir: Path) -> dict[str, Any]:
    script_figures_dir = script_output_dir / "figures"
    notebook_figures_dir = notebook_output_dir / "figures"
    if not script_figures_dir.exists() or not notebook_figures_dir.exists():
        return {
            "file": "figures",
            "kind": "figures",
            "status": "FAIL",
            "failure_type": "missing_file",
            "detail": (
                f"missing figures directory; script_exists={script_figures_dir.exists()}, "
                f"notebook_exists={notebook_figures_dir.exists()}"
            ),
        }
    script_figures = sorted(path.name for path in script_figures_dir.glob("*.png"))
    notebook_figures = sorted(path.name for path in notebook_figures_dir.glob("*.png"))
    return {
        "file": "figures",
        "kind": "figures",
        "status": "PASS" if script_figures == notebook_figures else "FAIL",
        "failure_type": None if script_figures == notebook_figures else "figure_names",
        "detail": f"script={len(script_figures)}, notebook={len(notebook_figures)}",
        "script_figures": script_figures,
        "notebook_figures": notebook_figures,
    }


def print_equivalence_summary(report: dict[str, Any]) -> None:
    print("Script vs Notebook Equivalence Check")
    print("------------------------------------")
    labels = [
        ("File structure", "file_structure"),
        ("Row counts", "row_counts"),
        ("Column structure", "column_structure"),
        ("Categorical fields", "categorical_fields"),
        ("Final selected method", "final_selected_method"),
        ("Final configuration", "final_configuration"),
        ("Numeric metrics within tolerance", "numeric_metrics"),
        ("Environment provenance recorded", "environment_provenance"),
    ]
    for label, key in labels:
        print(f"{label}: {report['summary'][key]}")

    maximum = report.get("maximum_numeric_difference")
    if maximum is not None:
        print()
        print("Maximum observed numeric difference:")
        print(
            f"{maximum['file']} / {maximum['column']} row {maximum['row']} = "
            f"{maximum['absolute_difference']:.12g}"
        )

    print()
    print(f"Overall equivalence: {'PASS' if report['all_passed'] else 'FAIL'}")
    if not report["all_passed"]:
        print()
        for mismatch in _format_failures(report["records"]):
            print(mismatch)


def format_equivalence_failures(report: dict[str, Any]) -> list[str]:
    return _format_failures(report["records"])


def _compare_numeric_column(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    left_values = left[column].to_numpy(dtype=float)
    right_values = right[column].to_numpy(dtype=float)
    both_nan = np.isnan(left_values) & np.isnan(right_values)
    differences = np.abs(left_values - right_values)
    differences[both_nan] = 0.0
    scale = _numeric_scale(left, right, column)
    allowed = NUMERIC_ATOL + NUMERIC_RTOL * scale
    matches = (differences <= allowed) | both_nan

    maximum = None
    if len(differences):
        max_index = int(np.nanargmax(differences))
        maximum = _numeric_detail(
            column,
            max_index,
            left_values[max_index],
            right_values[max_index],
            differences[max_index],
            allowed[max_index],
        )

    mismatches = []
    for row in np.flatnonzero(~matches)[:MAX_MISMATCH_EXAMPLES]:
        row_index = int(row)
        mismatches.append(
            _numeric_detail(
                column,
                row_index,
                left_values[row_index],
                right_values[row_index],
                differences[row_index],
                allowed[row_index],
            )
        )
    return mismatches, maximum


def _numeric_scale(left: pd.DataFrame, right: pd.DataFrame, column: str) -> np.ndarray:
    scale = np.maximum(
        np.nan_to_num(np.abs(left[column].to_numpy(dtype=float)), nan=0.0),
        np.nan_to_num(np.abs(right[column].to_numpy(dtype=float)), nan=0.0),
    )
    if column.startswith("delta_"):
        base = column[len("delta_") :]
        for suffix in ("before", "after"):
            paired_column = f"{base}_{suffix}"
            if paired_column in left.columns and paired_column in right.columns:
                scale = np.maximum(
                    scale,
                    np.nan_to_num(np.abs(left[paired_column].to_numpy(dtype=float)), nan=0.0),
                )
                scale = np.maximum(
                    scale,
                    np.nan_to_num(np.abs(right[paired_column].to_numpy(dtype=float)), nan=0.0),
                )
    return scale


def _compare_exact_column(left: pd.DataFrame, right: pd.DataFrame, column: str) -> dict[str, Any] | None:
    script_values = left[column].fillna("<NA>").astype(str)
    notebook_values = right[column].fillna("<NA>").astype(str)
    mismatched_rows = np.flatnonzero((script_values != notebook_values).to_numpy())
    if len(mismatched_rows) == 0:
        return None
    row = int(mismatched_rows[0])
    return {
        "column": column,
        "row": row,
        "script_value": script_values.iloc[row],
        "notebook_value": notebook_values.iloc[row],
    }


def _compare_exact_json(name: str, left: Any, right: Any) -> dict[str, Any]:
    mismatches = _compare_json_values(left, right, exact_numeric=True)
    return {
        "file": name,
        "kind": "json",
        "status": "PASS" if not mismatches else "FAIL",
        "failure_type": None if not mismatches else "json_exact",
        "detail": "matched exactly after runtime keys were ignored" if not mismatches else "JSON exact mismatch",
        "mismatches": mismatches[:MAX_MISMATCH_EXAMPLES],
    }


def _compare_tolerant_json(name: str, left: Any, right: Any) -> dict[str, Any]:
    mismatches = _compare_json_values(left, right, exact_numeric=False)
    maximum = _maximum_json_numeric_difference(left, right)
    first = mismatches[0] if mismatches else None
    return {
        "file": name,
        "kind": "json",
        "status": "PASS" if not mismatches else "FAIL",
        "failure_type": None if not mismatches else first.get("type", "json"),
        "detail": (
            "matched within numeric tolerance after runtime keys were ignored"
            if not mismatches
            else f"JSON mismatch at {first['path']}"
        ),
        "mismatches": mismatches[:MAX_MISMATCH_EXAMPLES],
        "max_numeric_difference": maximum,
    }


def _compare_environment_report(name: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if set(left) != set(right):
        return {
            "file": name,
            "kind": "json",
            "status": "FAIL",
            "failure_type": "json_structure",
            "detail": "environment report key mismatch",
            "script_keys": sorted(left),
            "notebook_keys": sorted(right),
        }

    exact_mismatches = []
    provenance_differences = {}
    for key in sorted(left):
        if key in ENVIRONMENT_PROVENANCE_KEYS:
            if left[key] != right[key]:
                provenance_differences[key] = {
                    "script_value": left[key],
                    "notebook_value": right[key],
                }
        elif left[key] != right[key]:
            exact_mismatches.append(
                {"path": f"/{key}", "script_value": left[key], "notebook_value": right[key]}
            )

    if exact_mismatches:
        return {
            "file": name,
            "kind": "json",
            "status": "FAIL",
            "failure_type": "json_exact",
            "detail": "environment capability mismatch",
            "mismatches": exact_mismatches[:MAX_MISMATCH_EXAMPLES],
        }
    return {
        "file": name,
        "kind": "json",
        "status": "PASS",
        "detail": "environment provenance recorded; version differences do not fail equivalence",
        "provenance_differences": provenance_differences,
    }


def _compare_json_values(
    left: Any,
    right: Any,
    path: str = "",
    exact_numeric: bool = False,
) -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        mismatches = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}/{key}"
            if key not in left or key not in right:
                mismatches.append({"type": "json_structure", "path": child_path, "detail": "missing key"})
            else:
                mismatches.extend(_compare_json_values(left[key], right[key], child_path, exact_numeric))
        return mismatches
    if isinstance(left, list) and isinstance(right, list):
        mismatches = []
        if len(left) != len(right):
            mismatches.append(
                {
                    "type": "json_structure",
                    "path": path,
                    "detail": f"list length mismatch; script={len(left)}, notebook={len(right)}",
                }
            )
        for index, (script_value, notebook_value) in enumerate(zip(left, right)):
            mismatches.extend(
                _compare_json_values(script_value, notebook_value, f"{path}[{index}]", exact_numeric)
            )
        return mismatches
    if _is_number(left) and _is_number(right) and not exact_numeric:
        return [] if _numbers_close(left, right) else [_json_numeric_mismatch(path, left, right)]
    if left != right and not (_is_nan(left) and _is_nan(right)):
        return [
            {
                "type": "json_exact",
                "path": path,
                "script_value": _json_safe(left),
                "notebook_value": _json_safe(right),
            }
        ]
    return []


def _numbers_close(left: int | float, right: int | float) -> bool:
    if _is_nan(left) and _is_nan(right):
        return True
    difference = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)))
    return difference <= NUMERIC_ATOL + NUMERIC_RTOL * scale


def _json_numeric_mismatch(path: str, left: int | float, right: int | float) -> dict[str, Any]:
    difference = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)))
    allowed = NUMERIC_ATOL + NUMERIC_RTOL * scale
    return {
        "type": "numeric",
        "path": path,
        "script_value": _json_safe(left),
        "notebook_value": _json_safe(right),
        "absolute_difference": difference,
        "allowed_tolerance": allowed,
    }


def _maximum_json_numeric_difference(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        maximum = None
        for key in sorted(set(left) & set(right)):
            candidate = _maximum_json_numeric_difference(left[key], right[key], f"{path}/{key}")
            if _is_larger_numeric_difference(candidate, maximum):
                maximum = candidate
        return maximum
    if isinstance(left, list) and isinstance(right, list):
        maximum = None
        for index, (script_value, notebook_value) in enumerate(zip(left, right)):
            candidate = _maximum_json_numeric_difference(script_value, notebook_value, f"{path}[{index}]")
            if _is_larger_numeric_difference(candidate, maximum):
                maximum = candidate
        return maximum
    if _is_number(left) and _is_number(right):
        if _is_nan(left) and _is_nan(right):
            return None
        return {
            "path": path,
            "absolute_difference": abs(float(left) - float(right)),
            "script_value": _json_safe(left),
            "notebook_value": _json_safe(right),
        }
    return None


def _maximum_numeric_difference(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    maximum = None
    for record in records:
        candidate = record.get("max_numeric_difference")
        if candidate is None:
            continue
        if "column" not in candidate:
            candidate = {
                "file": record["file"],
                "column": candidate.get("path", "<json>"),
                "row": "<json>",
                **candidate,
            }
        else:
            candidate = {"file": record["file"], **candidate}
        if _is_larger_numeric_difference(candidate, maximum):
            maximum = candidate
    return maximum


def _build_summary(records: list[dict[str, Any]], final_selection: dict[str, Any]) -> dict[str, str]:
    failure_types = {record.get("failure_type") for record in records if record["status"] == "FAIL"}
    final_config = next(
        (record for record in records if record["file"] == "final_selected_configuration.json"),
        {"status": "FAIL"},
    )
    environment = next(
        (record for record in records if record["file"] == "environment_report.json"),
        {"status": "FAIL"},
    )
    return {
        "file_structure": _pass_fail("missing_file" not in failure_types and "figure_names" not in failure_types),
        "row_counts": _pass_fail("row_count" not in failure_types),
        "column_structure": _pass_fail("columns" not in failure_types and "json_structure" not in failure_types),
        "categorical_fields": _pass_fail(not (failure_types & {"categorical", "json_exact"})),
        "final_selected_method": _pass_fail(final_selection["status"] == "PASS"),
        "final_configuration": _pass_fail(final_config["status"] == "PASS"),
        "numeric_metrics": _pass_fail("numeric" not in failure_types),
        "environment_provenance": _pass_fail(environment["status"] == "PASS"),
    }


def _final_selection_summary(script_output_dir: Path, notebook_output_dir: Path) -> dict[str, Any]:
    script_config = json.loads((script_output_dir / "final_selected_configuration.json").read_text(encoding="utf-8"))
    notebook_config = json.loads((notebook_output_dir / "final_selected_configuration.json").read_text(encoding="utf-8"))
    script_method = script_config.get("selected_method")
    notebook_method = notebook_config.get("selected_method")
    return {
        "status": "PASS" if script_method == notebook_method else "FAIL",
        "script_selected_method": script_method,
        "notebook_selected_method": notebook_method,
    }


def _format_failures(records: list[dict[str, Any]]) -> list[str]:
    failures = []
    for record in records:
        if record["status"] != "FAIL":
            continue
        failures.append(f"Mismatch: File: {record['file']}; {record.get('detail', 'failed')}")
        for mismatch in record.get("mismatches", [])[:MAX_MISMATCH_EXAMPLES]:
            if mismatch.get("type") == "numeric" or "absolute_difference" in mismatch:
                failures.append(
                    "  "
                    f"Column/Path: {mismatch.get('column', mismatch.get('path'))}; "
                    f"Row: {mismatch.get('row', '<json>')}; "
                    f"Script: {mismatch.get('script_value')}; "
                    f"Notebook: {mismatch.get('notebook_value')}; "
                    f"Absolute difference: {mismatch.get('absolute_difference')}; "
                    f"Allowed tolerance: {mismatch.get('allowed_tolerance')}"
                )
            else:
                failures.append(
                    "  "
                    f"Column/Path: {mismatch.get('column', mismatch.get('path'))}; "
                    f"Row: {mismatch.get('row', '<json>')}; "
                    f"Script: {mismatch.get('script_value')}; "
                    f"Notebook: {mismatch.get('notebook_value')}"
                )
    return failures


def _numeric_detail(
    column: str,
    row: int,
    script_value: float,
    notebook_value: float,
    difference: float,
    allowed: float,
) -> dict[str, Any]:
    return {
        "column": column,
        "row": row,
        "script_value": _json_safe(script_value),
        "notebook_value": _json_safe(notebook_value),
        "absolute_difference": float(difference),
        "allowed_tolerance": float(allowed),
    }


def _is_larger_numeric_difference(candidate: dict[str, Any] | None, current: dict[str, Any] | None) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return candidate.get("absolute_difference", -math.inf) > current.get("absolute_difference", -math.inf)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if _is_nan(value):
        return "NaN"
    return value


def _pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"
