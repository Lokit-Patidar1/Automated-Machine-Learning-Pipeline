import pandas as pd

def basic_summary(df: pd.DataFrame) -> dict:
	return {
		'shape': df.shape,
		'numeric_cols': len(df.select_dtypes(include=['number']).columns),
		'categorical_cols': len(df.select_dtypes(include=['object']).columns),
		'head_html': df.head(10).to_html(classes='table table-striped', index=False),
		'describe_html': df.describe().to_html(classes='table table-striped'),
		'dtypes_html': df.dtypes.to_frame('Data Type').to_html(classes='table table-striped'),
	}


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
	missing = df.isnull().sum()
	missing = missing[missing > 0].sort_values(ascending=False)
	return pd.DataFrame({
		'Column': missing.index,
		'Missing Count': missing.values,
		'Missing %': (missing.values / len(df) * 100).round(2)
	}).reset_index(drop=True)
