"""
Azure ML entrypoint for the Lighthouse (lh_v2) training pipeline.

Runs the full pipeline (driver analysis -> train/validate -> forecast -> reconcile)
and writes artifacts + a manifest.json into --output-dir so the static HTML portal
can consume them.

Artifacts written:
  selected_drivers.csv
  validation.csv
  account_forecasts.csv
  account_forecasts_reconciled.csv
  charts/<account>.json         # actuals + forecast series (interactive chart)
  charts/<account>.png          # static chart
  audit/config.resolved.yml
  audit/run.json
  manifest.json                 # entry point for the portal

All paths come from CLI args so AML can mount data assets and the outputs folder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib as pth
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np
import polars as pl

import lh_v2
from lh_v2.account_reconciliation import apply_account_reconciliation
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.forecasting.account_forecasting.model_forecasting.forecast_accounts import (
    create_account_forecasts,
)
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.train_models import (
    train_and_validate_models,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.io.output.output_data import (
    save_account_forecasts_csv,
    save_account_validation_csv,
    save_selected_drivers_csv,
)
from lh_v2.params import parse_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Lighthouse training entrypoint')
    p.add_argument('--accounts', required=True, help='Path to accounts CSV (uri_file mount)')
    p.add_argument('--drivers', required=True, help='Path to drivers CSV (uri_file mount)')
    p.add_argument('--config', required=True, help='Path to config.yml (uri_file mount)')
    p.add_argument('--output-dir', required=True, help='Directory to write artifacts into')
    p.add_argument('--run-id', default=os.environ.get('AZUREML_RUN_ID', ''))
    return p.parse_args()


def _hash_file(path: pth.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _float_series_for_plot(values: list[float | None]) -> list[float]:
    """Matplotlib ``plot`` expects numeric y; map missing values to NaN."""
    return [float('nan') if v is None else float(v) for v in values]


def _dates_for_plot(dates: list[dt.date]) -> np.ndarray[Any, Any]:
    """Matplotlib stubs reject ``list[date]`` as *args; use datetime64 ndarray."""
    return np.array([np.datetime64(d.isoformat()) for d in dates])


def _write_chart(
    out_dir: pth.Path,
    account: str,
    dates: list[dt.date],
    actuals: list[float | None],
    forecast: list[float | None],
    reconciled: list[float | None] | None = None,
) -> dict[str, Any]:
    chart_dir = out_dir / 'charts'
    chart_dir.mkdir(parents=True, exist_ok=True)
    safe = account.replace(' ', '_').replace('/', '_')

    data = {
        'account': account,
        'dates': [d.isoformat() for d in dates],
        'actual': actuals,
        'forecast': forecast,
        'reconciled': reconciled,
    }
    json_path = chart_dir / f'{safe}.json'
    json_path.write_text(json.dumps(data))

    png_path: pth.Path | None = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        xd = _dates_for_plot(dates)
        ax.plot(xd, _float_series_for_plot(actuals), label='actual', linewidth=2)
        ax.plot(xd, _float_series_for_plot(forecast), label='forecast', linewidth=2, linestyle='--')
        if reconciled is not None and any(v is not None for v in reconciled):
            ax.plot(
                xd,
                _float_series_for_plot(reconciled),
                label='reconciled',
                linewidth=2,
                linestyle=':',
            )
        ax.set_title(f'{account}')
        ax.set_xlabel('date')
        ax.set_ylabel('value')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        png_path = chart_dir / f'{safe}.png'
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
    except Exception as exc:  # charting must never kill the run
        print(f'[chart] skipped PNG for {account}: {exc}', file=sys.stderr)

    return {
        'account': account,
        'json': f'charts/{safe}.json',
        'png': f'charts/{safe}.png' if png_path else None,
    }


def _build_series_per_account(
    accounts_df_path: pth.Path,
    forecasts_df_path: pth.Path,
    reconciled_df_path: pth.Path | None,
) -> dict[str, dict[str, Any]]:
    """
    Stitch actuals (demo accounts CSV) with forecasts on a common date axis.
    Returns: { account: { dates:[...], actuals:[...], forecast:[...], reconciled:[...] } }
    """
    actuals = pl.read_csv(accounts_df_path)
    actuals = actuals.with_columns(pl.col('period_key').str.to_date())
    actuals_grouped = (
        actuals.group_by(['period_key', 'account_key']).agg(pl.col('value').sum())
    )

    fc = pl.read_csv(forecasts_df_path)
    fc = fc.with_columns(pl.col('date').str.to_date())
    fc_grouped = fc.group_by(['date', 'account_key']).agg(pl.col('forecast_value').sum())

    rec_grouped: pl.DataFrame | None = None
    if reconciled_df_path and reconciled_df_path.exists():
        rec = pl.read_csv(reconciled_df_path)
        rec = rec.with_columns(pl.col('date').str.to_date())
        rec_grouped = rec.group_by(['date', 'account_key']).agg(
            pl.col('forecast_value').sum()
        )

    out: dict[str, dict[str, Any]] = {}
    accounts = sorted(set(fc_grouped['account_key'].unique().to_list()))
    for acc in accounts:
        a_df = actuals_grouped.filter(pl.col('account_key') == acc).rename(
            {'period_key': 'date', 'value': 'actual'}
        )
        f_df = fc_grouped.filter(pl.col('account_key') == acc).rename(
            {'forecast_value': 'forecast'}
        )
        joined = a_df.join(f_df, on=['date', 'account_key'], how='full', coalesce=True)
        if rec_grouped is not None:
            r_df = rec_grouped.filter(pl.col('account_key') == acc).rename(
                {'forecast_value': 'reconciled'}
            )
            joined = joined.join(
                r_df, on=['date', 'account_key'], how='full', coalesce=True
            )
        joined = joined.sort('date')
        dates = joined['date'].to_list()
        actuals_vals = joined['actual'].to_list() if 'actual' in joined.columns else []
        forecast_vals = joined['forecast'].to_list() if 'forecast' in joined.columns else []
        reconciled_vals = (
            joined['reconciled'].to_list() if 'reconciled' in joined.columns else None
        )
        out[acc] = {
            'dates': dates,
            'actuals': actuals_vals,
            'forecast': forecast_vals,
            'reconciled': reconciled_vals,
        }
    return out


def main() -> int:
    args = parse_args()
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()

    out_dir = pth.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'audit').mkdir(parents=True, exist_ok=True)

    config_path = pth.Path(args.config)
    accounts_path = pth.Path(args.accounts)
    drivers_path = pth.Path(args.drivers)

    print(f'lh_v2 @ {getattr(lh_v2, "__file__", "?")}')
    print(f'accounts: {accounts_path}')
    print(f'drivers : {drivers_path}')
    print(f'config  : {config_path}')
    print(f'output  : {out_dir}')

    lh_params = parse_yaml(config_path)

    shutil.copyfile(config_path, out_dir / 'audit' / 'config.resolved.yml')

    dr_data = load_data_driver_ranking(
        lh_params=lh_params,
        account_source=accounts_path,
        driver_source=drivers_path,
    )

    analysis_results = analyze_drivers_full(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
        output_params=lh_params.output_params,
    )

    save_selected_drivers_csv(
        da_output=analysis_results,
        segment_type=lh_params.segment,
        region_type=lh_params.region,
        output_pth=out_dir / 'selected_drivers',
    )

    training_input = ModelTrainingInput(
        accounts_drivers=dr_data.select_drivers(analysis_results.selected_drivers),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
    )

    training_results = train_and_validate_models(
        model_training_info=training_input,
        general_params=lh_params.general_params,
        af_params=lh_params.account_forecast_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

    save_account_validation_csv(
        validation_output=training_results,
        actuals=dr_data.accounts,
        validation_daterange=(
            lh_params.general_params.validation_start_date,
            lh_params.general_params.validation_end_date,
        ),
        output_pth=out_dir / 'validation',
    )

    forecasting_input = ModelForecastingInput(
        accounts_drivers=dr_data.select_drivers(analysis_results.selected_drivers),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
        selected_model=training_results.select_forecast_methods(),
        best_params=training_results.format_best_params(),
        validation_errors=training_results.get_selected_methods_errors(
            selected_methods=training_results.select_forecast_methods(),
            actuals=dr_data.accounts,
            val_date_range=(
                lh_params.general_params.validation_start_date,
                lh_params.general_params.validation_end_date,
            ),
        ),
    )

    forecasting_results = create_account_forecasts(
        forecasting_input=forecasting_input,
        general_params=lh_params.general_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

    save_account_forecasts_csv(
        account_forecasts=forecasting_results,
        output_pth=out_dir / 'account_forecasts',
    )

    reconciled_results = apply_account_reconciliation(
        forecasting_data=forecasting_results,
        reconciliation_params=lh_params.account_reconciliation_params,
        output_params=lh_params.output_params,
    )

    save_account_forecasts_csv(
        account_forecasts=reconciled_results,
        output_pth=out_dir / 'account_forecasts_reconciled',
    )

    series = _build_series_per_account(
        accounts_df_path=accounts_path,
        forecasts_df_path=out_dir / 'account_forecasts.csv',
        reconciled_df_path=out_dir / 'account_forecasts_reconciled.csv',
    )

    chart_entries: list[dict[str, Any]] = []
    for account, s in series.items():
        chart_entries.append(
            _write_chart(
                out_dir=out_dir,
                account=account,
                dates=s['dates'],
                actuals=s['actuals'],
                forecast=s['forecast'],
                reconciled=s['reconciled'],
            )
        )

    elapsed = time.time() - t0
    ended = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f'local-{started:%Y%m%d%H%M%S}'

    run_info = {
        'run_id': run_id,
        'started_at': started.isoformat(),
        'ended_at': ended.isoformat(),
        'elapsed_seconds': round(elapsed, 3),
        'python_version': sys.version,
        'lh_v2_path': getattr(lh_v2, '__file__', None),
        'inputs': {
            'accounts': {
                'path': str(accounts_path),
                'sha256': _hash_file(accounts_path),
            },
            'drivers': {
                'path': str(drivers_path),
                'sha256': _hash_file(drivers_path),
            },
            'config': {
                'path': str(config_path),
                'sha256': _hash_file(config_path),
            },
        },
        'segment': str(lh_params.segment),
        'region': str(lh_params.region),
        'accounts': [str(a) for a in lh_params.accounts],
        'training_daterange': [
            lh_params.general_params.training_start_date.isoformat(),
            lh_params.general_params.training_end_date.isoformat(),
        ],
        'validation_daterange': [
            lh_params.general_params.validation_start_date.isoformat(),
            lh_params.general_params.validation_end_date.isoformat(),
        ],
        'testing_daterange': [
            lh_params.general_params.testing_start_date.isoformat(),
            lh_params.general_params.testing_end_date.isoformat(),
        ],
    }
    (out_dir / 'audit' / 'run.json').write_text(json.dumps(run_info, indent=2, default=str))

    manifest = {
        'schema_version': '1',
        'run_id': run_id,
        'generated_at': ended.isoformat(),
        'segment': run_info['segment'],
        'region': run_info['region'],
        'accounts': run_info['accounts'],
        'daterange': {
            'training': run_info['training_daterange'],
            'validation': run_info['validation_daterange'],
            'testing': run_info['testing_daterange'],
        },
        'tables': {
            'selected_drivers': 'selected_drivers.csv',
            'validation': 'validation.csv',
            'account_forecasts': 'account_forecasts.csv',
            'account_forecasts_reconciled': 'account_forecasts_reconciled.csv',
        },
        'charts': chart_entries,
        'audit': {
            'config_resolved': 'audit/config.resolved.yml',
            'run': 'audit/run.json',
        },
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

    print('-' * 60)
    print(f'Done in {elapsed:.1f}s')
    print(f'Manifest: {out_dir / "manifest.json"}')
    print('Artifacts:')
    for p in sorted(out_dir.rglob('*')):
        if p.is_file():
            print('  ', p.relative_to(out_dir))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
