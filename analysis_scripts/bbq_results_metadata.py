import json
import os
import glob
import pandas as pd
from typing import List, Dict, Any

UNKNOWN_ANSWERS = [
    "unknown",
    "cannot be determined",
    "can't be determined",
    "not answerable",
    "not known",
    "not enough info",
    "not enough information",
    "cannot answer",
    "can't answer",
    "undetermined",
]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a jsonl file into a list of dictionaries."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def generate_additional_metadata(data_dir: str = "../data", output_csv: str = "additional_metadata.csv") -> pd.DataFrame:
    """Generate a simplified additional metadata file from the BBQ data files."""
    records: List[Dict[str, Any]] = []
    for file in glob.glob(os.path.join(data_dir, "*.jsonl")):
        data = load_jsonl(file)
        for ex in data:
            ans0_text, ans0_info = ex["answer_info"]["ans0"]
            ans1_text, ans1_info = ex["answer_info"]["ans1"]
            ans2_text, ans2_info = ex["answer_info"]["ans2"]
            stereo_groups = ex.get("additional_metadata", {}).get("stereotyped_groups", [])
            target_loc = None
            for idx, info in enumerate([ans0_info, ans1_info, ans2_info]):
                if any(str(info).lower() == str(sg).lower() for sg in stereo_groups):
                    target_loc = idx
                    break
            label_type = "name" if int(ex["question_index"]) > 25 else "label"
            records.append({
                "category": ex.get("category"),
                "question_index": ex.get("question_index"),
                "example_id": ex.get("example_id"),
                "target_loc": target_loc,
                "label_type": label_type,
                "Known_stereotyped_race": None,
                "Known_stereotyped_var2": None,
                "Relevant_social_values": ex.get("additional_metadata", {}).get("subcategory"),
                "corr_ans_aligns_var2": None,
                "corr_ans_aligns_race": None,
                "full_cond": None,
                "Known_stereotyped_groups": ";".join(stereo_groups),
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    generate_additional_metadata()
