from typing import Dict, Any, List
import pandas as pd

from .machine_learning import train_and_evaluate


def run_pipeline(df: pd.DataFrame, target_col: str, problem_type: str, selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
	return train_and_evaluate(df, target_col, problem_type, selected_models)
