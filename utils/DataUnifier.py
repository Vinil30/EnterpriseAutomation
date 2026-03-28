import pandas as pd
import json
import os
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TOOLS
# ============================================================================

@tool
def read_csv_preview(file_path: str, rows: int = 10) -> str:
    """Used to read CSV"""
    try:
        df = pd.read_csv(file_path, nrows=rows)

        info = {
            "file": file_path,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict(orient="records"),
            "null_counts": df.isnull().sum().to_dict(),
            "row_count_preview": len(df)
        }

        return json.dumps(info, indent=2, default=str)

    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"


@tool
def suggest_joins(files_info: str) -> str:
    """Suggested Joins"""
    try:
        files = json.loads(files_info)

        suggestions = []
        for f1, cols1 in files.items():
            for f2, cols2 in files.items():
                if f1 < f2:
                    common = list(set(cols1) & set(cols2))
                    if common:
                        suggestions.append(f"{f1} ↔ {f2} → {common}")

        return "\n".join(suggestions) if suggestions else "No common keys found."

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def merge_datasets(left_file: str, right_file: str, left_key: str, right_key: str, how: str = "left") -> str:
    """Merging operations"""
    try:
        left_df = pd.read_csv(left_file)
        right_df = pd.read_csv(right_file)

        merged = left_df.merge(right_df, left_on=left_key, right_on=right_key, how=how)

        output_path = f"merged_{os.path.basename(left_file).replace('.csv','')}_{os.path.basename(right_file)}"
        merged.to_csv(output_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": output_path,
            "rows": len(merged),
            "columns": list(merged.columns)
        })

    except Exception as e:
        return f"Error merging: {str(e)}"


@tool
def finalize_unified_dataset(df_path: str, selected_columns: str = None) -> str:
    """Unifying dataset"""
    try:
        df = pd.read_csv(df_path)

        if selected_columns:
            columns = json.loads(selected_columns)
            df = df[[c for c in columns if c in df.columns]]

        output_path = "unified_dataset.csv"
        df.to_csv(output_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": output_path,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        })

    except Exception as e:
        return f"Error finalizing: {str(e)}"


# ============================================================================
# MAIN PIPELINE CLASS (NO AGENT)
# ============================================================================

class SchemaUnificationPipeline:

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0
        )

    def unify(self, file_paths: List[str]) -> Dict[str, Any]:

        previews = {}

        # Step 1: Read previews
        for path in file_paths:
            preview = read_csv_preview.invoke({"file_path": path})
            parsed = json.loads(preview)
            previews[path] = parsed["columns"]

        # Step 2: Show join suggestions
        joins = suggest_joins.invoke({"files_info": json.dumps(previews)})
        print("\n🔗 Join Suggestions:\n", joins)

        # Step 3: Start with first file
        current_file = file_paths[0]

        # Step 4: Sequential merge
        for next_file in file_paths[1:]:

            # update previews dynamically
            for f in [current_file, next_file]:
                if f not in previews:
                    preview = read_csv_preview.invoke({"file_path": f})
                    parsed = json.loads(preview)
                    previews[f] = parsed["columns"]

            common_cols = list(set(previews[current_file]) & set(previews[next_file]))

            if not common_cols:
                print(f"⚠️ No common columns between {current_file} and {next_file}")
                continue

            key = common_cols[0]

            print(f"➡️ Merging {current_file} + {next_file} on '{key}'")

            result = merge_datasets.invoke({
                "left_file": current_file,
                "right_file": next_file,
                "left_key": key,
                "right_key": key,
                "how": "left"
            })

            result = json.loads(result)
            current_file = result["output_file"]

        # Step 5: Finalize
        final = finalize_unified_dataset.invoke({
            "df_path": current_file,
            "selected_columns": None
        })

        return json.loads(final)

