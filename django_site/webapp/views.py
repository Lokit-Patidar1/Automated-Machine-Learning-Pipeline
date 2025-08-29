import io
import pickle
from typing import Dict, Any

from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse, FileResponse
from django.core.files.storage import default_storage
from django.conf import settings
import pandas as pd

from core_ml.main import run_pipeline
from core_ml.eda import basic_summary
from core_ml.serialization import serialize_model

SESSION_DF_KEY = 'uploaded_df'
SESSION_RESULTS_KEY = 'ml_results'


def _read_df_from_session(request: HttpRequest) -> pd.DataFrame | None:
	csv_path = request.session.get(SESSION_DF_KEY)
	if not csv_path:
		return None
	try:
		return pd.read_csv(csv_path)
	except Exception:
		return None


def upload_view(request: HttpRequest) -> HttpResponse:
	if request.method == 'POST' and request.FILES.get('file'):
		f = request.FILES['file']
		path = default_storage.save(f"uploads/{f.name}", f)
		abs_path = settings.MEDIA_ROOT / path if hasattr(settings, 'MEDIA_ROOT') else path
		df = pd.read_csv(abs_path) if f.name.endswith('.csv') else pd.read_excel(abs_path)
		cache_csv = settings.MEDIA_ROOT / 'uploads' / 'sourcedata.csv'
		cache_csv.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(cache_csv, index=False)
		request.session[SESSION_DF_KEY] = str(cache_csv)
		return redirect('eda')
	return render(request, 'upload.html')


def eda_view(request: HttpRequest) -> HttpResponse:
	df = _read_df_from_session(request)
	if df is None:
		return redirect('upload')
	context = basic_summary(df)
	context['columns'] = df.columns
	return render(request, 'eda.html', context)


def train_view(request: HttpRequest) -> HttpResponse:
	df = _read_df_from_session(request)
	if df is None:
		return redirect('upload')
	context: Dict[str, Any] = {'df_columns': df.columns}
	if request.method == 'POST':
		target_col = request.POST.get('target_col')
		problem_type = request.POST.get('problem_type')
		selected_models = request.POST.getlist('models')
		results = run_pipeline(df, target_col, problem_type, selected_models)
		if not results:
			context['error'] = 'Training failed or returned no results.'
		else:
			request.session[SESSION_RESULTS_KEY] = pickle.dumps(results).hex()
			context['results'] = {m: {k: v for k, v in res.items() if k not in ['Report', 'Model', 'Scaler']} for m, res in results.items()}
	return render(request, 'train.html', context)


def download_view(request: HttpRequest) -> HttpResponse:
	results_hex = request.session.get(SESSION_RESULTS_KEY)
	if not results_hex:
		return redirect('train')
	results: Dict[str, Dict[str, Any]] = pickle.loads(bytes.fromhex(results_hex))
	if request.GET.get('all') == '1':
		buffer = io.BytesIO()
		package = {}
		for model_name, data in results.items():
			package[model_name] = {
				'model': data.get('Model'),
				'scaler': data.get('Scaler'),
				'metrics': {k: v for k, v in data.items() if k not in ['Model', 'Scaler', 'Report']}
			}
		pickle.dump(package, buffer)
		buffer.seek(0)
		return FileResponse(buffer, as_attachment=True, filename='automl_models_package.pkl')
	model_name = request.GET.get('name')
	if not model_name or model_name not in results:
		return redirect('train')
	blob = serialize_model(results[model_name].get('Model'), results[model_name].get('Scaler'), model_name)
	return HttpResponse(blob, content_type='application/octet-stream', headers={'Content-Disposition': f'attachment; filename={model_name.replace(" ", "_").lower()}_model.pkl'})
