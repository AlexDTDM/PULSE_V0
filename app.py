import contextlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import date, time, timedelta, datetime

import dateparser
import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── Database connection ───────────────────────────────────────────────────────

ENGINE = create_engine(os.getenv("DATABASE_URL"))

# Postgres lowercases column names — map them back to what the code expects
_COL_MAP = {
    "timestamp_answer": "Timestamp_answer",
    "participantsex":   "participantSex",
    "interaction_id":   "interaction_ID",
}

def _read_table(table: str) -> pd.DataFrame:
    """Read a DB table and restore original column casing."""
    df = pd.read_sql(f"SELECT * FROM {table}", ENGINE)
    return df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})

def _write_table(df: pd.DataFrame, table: str, **kwargs):
    """Write to DB with lowercase column names (Postgres convention)."""
    df = df.rename(columns={v: k for k, v in _COL_MAP.items()})
    df.columns = df.columns.str.lower()
    df.to_sql(table, ENGINE, **kwargs)


# ── Helper functions (from process_data.ipynb) ────────────────────────────────

def compute_p_distribution(df, BIN_HOURS):
    df['day_of_week'] = df['Timestamp_answer'].dt.dayofweek
    df['hour'] = df['Timestamp_answer'].dt.hour
    df['hour_bin'] = (df['hour'] // BIN_HOURS) * BIN_HOURS
    df['week_slot'] = df['day_of_week'] * 24 + df['hour_bin']

    slot_counts = df.groupby(['day_of_week', 'hour_bin']).size().reset_index(name='count')
    slot_counts['probability'] = slot_counts['count'] / slot_counts['count'].sum()

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    slot_counts['label'] = slot_counts.apply(
        lambda r: f"{day_names[int(r['day_of_week'])]} {int(r['hour_bin']):02d}:00-{int(r['hour_bin']) + BIN_HOURS:02d}:00",
        axis=1
    )
    return slot_counts.sort_values(['day_of_week', 'hour_bin']).reset_index(drop=True)



def merge_demographics(df, df_demographic, age_bins=None, age_labels=None):
    if age_bins is None:
        age_bins   = [0, 40, 60, 120]
    if age_labels is None:
        age_labels = ['<40', '40-60', '60+']

    df_merged = df.merge(df_demographic, on='participant', how='left')
    df_merged['sex'] = df_merged['participantSex'].map({'female': 'f', 'male': 'm'})
    df_merged['age_range'] = pd.cut(
        df_merged['age'], bins=age_bins, labels=age_labels, right=False,
    ).astype(object).where(df_merged['age'].notna())

    return df_merged[['participant', 'Timestamp_answer', 'sex', 'age_range']]


def select_by_demographics(df_merged, sex=None, age=None, age_bins=None, age_labels=None):
    if age_bins is None:
        age_bins   = [0, 40, 60, 120]
    if age_labels is None:
        age_labels = ['<40', '40-60', '60+']

    df_out = df_merged.copy()

    if sex is not None:
        df_out = df_out[df_out['sex'] == sex]

    if age is not None:
        age_range = None
        for i in range(len(age_bins) - 1):
            if age_bins[i] <= age < age_bins[i + 1]:
                age_range = age_labels[i]
                break
        if age_range is None:
            raise ValueError(f"Age {age} is outside the defined bins {age_bins}.")
        df_out = df_out[df_out['age_range'] == age_range]

    return df_out.reset_index(drop=True)


@dataclass
class CohortResult:
    neighbor_indices: list
    active_categoricals: list
    active_categorical_values: dict
    pool_before: int
    pool_after_categorical: int
    pool_after_knn: int
    effective_k: int
    fallback: bool
    age_range: tuple = None


def select_tailored_cohort(
    df_demographic: pd.DataFrame,
    patient_id: str,
    categorical_cols: list = None,
    continuous_cols: list = None,
    categorical_priority: list = None,
    exclude_cols: list = None,
    k: int = 100,
    n_min: int = 100,
) -> CohortResult:
    if exclude_cols is None:
        exclude_cols = ["intevention"]

    if categorical_cols is None or continuous_cols is None:
        skip = {"participant"} | set(exclude_cols)
        feature_cols = [c for c in df_demographic.columns if c not in skip]
        auto_cat, auto_cont = [], []
        for col in feature_cols:
            vals = df_demographic[col].dropna()
            if vals.dtype == object:
                vals = vals.str.strip().replace("", pd.NA).dropna()
            n_unique = vals.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df_demographic[col])
            if n_unique == 2:
                auto_cat.append(col)
            elif is_numeric:
                auto_cont.append(col)
        if categorical_cols is None:
            categorical_cols = auto_cat
        if continuous_cols is None:
            continuous_cols = auto_cont

    if categorical_priority is None:
        categorical_priority = list(reversed(categorical_cols))

    patient_row = df_demographic[df_demographic["participant"] == patient_id]
    if patient_row.empty:
        raise ValueError(f"Patient '{patient_id}' not found in df_demographic")
    patient = patient_row.iloc[0]

    pool = df_demographic[df_demographic["participant"] != patient_id].copy()
    pool_before = len(pool)

    active_filters = [
        col for col in categorical_cols
        if col in patient.index and pd.notna(patient[col])
    ]
    active_filters = sorted(
        active_filters,
        key=lambda c: categorical_priority.index(c) if c in categorical_priority else len(categorical_priority)
    )

    def apply_filters(df, filters):
        mask = pd.Series(True, index=df.index)
        for col in filters:
            mask &= df[col] == patient[col]
        return df[mask]

    while True:
        filtered = apply_filters(pool, active_filters)
        if len(filtered) >= n_min or len(active_filters) == 0:
            break
        active_filters.pop(0)

    pool = filtered if len(filtered) > 0 else pool
    fallback = len(active_filters) == 0
    pool_after_categorical = len(pool)

    active_categorical_values = {col: patient[col] for col in active_filters}

    effective_k = min(k, len(pool))

    if effective_k == 0:
        return CohortResult(
            neighbor_indices=[], active_categoricals=active_filters,
            active_categorical_values=active_categorical_values,
            pool_before=pool_before, pool_after_categorical=pool_after_categorical,
            pool_after_knn=0, effective_k=0, fallback=True,
        )

    all_data = pd.concat([pool, patient_row], ignore_index=True)
    stds = {}
    for col in continuous_cols:
        if col in all_data.columns:
            s = all_data[col].astype(float).std()
            stds[col] = s if s > 0 else 1.0

    distances = np.zeros(len(pool))
    for i, (_, row) in enumerate(pool.iterrows()):
        d_sum, d_count = 0.0, 0
        for col in continuous_cols:
            p_val = patient.get(col)
            c_val = row.get(col)
            if pd.notna(p_val) and pd.notna(c_val) and col in stds:
                d_sum += abs(float(p_val) - float(c_val)) / stds[col]
                d_count += 1
        distances[i] = d_sum / d_count if d_count > 0 else float("inf")

    nearest_idx = np.argsort(distances)[:effective_k]
    neighbor_indices = pool.index[nearest_idx].tolist()

    neighbor_ages = df_demographic.loc[neighbor_indices, "age"].dropna()
    age_range = (int(neighbor_ages.min()), int(neighbor_ages.max())) if not neighbor_ages.empty else None

    return CohortResult(
        neighbor_indices=neighbor_indices, active_categoricals=active_filters,
        active_categorical_values=active_categorical_values,
        pool_before=pool_before, pool_after_categorical=pool_after_categorical,
        pool_after_knn=effective_k, effective_k=effective_k, fallback=fallback,
        age_range=age_range,
    )


def parse_target_date(text):
    DAY_START = time(0, 0)
    DAY_END   = time(23, 59)

    _NEXT_RANGE = re.compile(
        r'\bin the next\s+(\d+)\s*(day|days|week|weeks|hour|hours|min(?:ute)?s?)\b',
        re.IGNORECASE,
    )

    if not (text and str(text).strip()):
        tomorrow = date.today() + timedelta(days=1)
        return tomorrow, tomorrow, DAY_START, DAY_END

    raw = str(text).strip()
    m = _NEXT_RANGE.search(raw)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        now = datetime.now()
        if 'hour' in unit:
            end_dt = now + timedelta(hours=n)
            return now.date(), end_dt.date(), now.time().replace(second=0, microsecond=0), end_dt.time().replace(second=0, microsecond=0)
        if 'min' in unit:
            end_dt = now + timedelta(minutes=n)
            return now.date(), end_dt.date(), now.time().replace(second=0, microsecond=0), end_dt.time().replace(second=0, microsecond=0)
        end_date = date.today() + (timedelta(weeks=n) if 'week' in unit else timedelta(days=n))
        return date.today(), end_date, DAY_START, DAY_END

    dt = dateparser.parse(raw, settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False})
    if dt is None:
        raise ValueError(f"Could not parse target_date='{raw}'.")
    return dt.date(), dt.date(), DAY_START, DAY_END


def get_top_k_peaks(df_final, start_day, end_day, start_time, end_time, K):
    bins = sorted(df_final['hour_bin'].unique())
    bin_hours = int(bins[1] - bins[0]) if len(bins) > 1 else 24

    DAY_START, DAY_END = time(0, 0), time(23, 59)
    seen, valid_slots = set(), []
    current = start_day

    while current <= end_day:
        dow = current.weekday()
        t_start = start_time if current == start_day else DAY_START
        t_end   = end_time   if current == end_day   else DAY_END
        t_start_h = t_start.hour + t_start.minute / 60
        t_end_h   = t_end.hour   + t_end.minute   / 60

        for b in bins:
            key = (dow, b)
            if key not in seen and b < t_end_h and (b + bin_hours) > t_start_h:
                seen.add(key)
                valid_slots.append({'day_of_week': dow, 'hour_bin': b, 'date': current})

        current += timedelta(days=1)

    if not valid_slots:
        return df_final.iloc[0:0].copy()

    valid_df    = pd.DataFrame(valid_slots)
    df_filtered = df_final.merge(valid_df, on=['day_of_week', 'hour_bin'])
    if df_filtered.empty:
        return pd.DataFrame(columns=['timestamp', 'p_final'])
    df_top      = df_filtered.nlargest(K, 'p_final')[['date', 'hour_bin', 'p_final']].copy()
    now = datetime.now()
    df_top['timestamp'] = df_top.apply(
        lambda r: datetime.combine(r['date'], time(int(r['hour_bin']), 0)), axis=1
    )
    # Filter out timestamps in the past
    df_top = df_top[df_top['timestamp'] > now]
    return df_top[['timestamp', 'p_final']].reset_index(drop=True)


# ── plot_p_distribution_demographic (adapted from process_data.ipynb) ─────────

def plot_p_distribution_demographic(group_by: str = "sex", BIN_HOURS: int = 4):
    """
    Compute and return a Plotly figure comparing interaction probability
    distributions across demographic groups in the population.

    Adapted from process_data.ipynb — ipywidgets removed, returns figure directly.

    Parameters
    ----------
    group_by  : 'sex' or 'age_range'
    BIN_HOURS : time bin size in hours
    """
    import plotly.graph_objects as go
    import plotly.colors

    df           = _read_table("interactions").assign(Timestamp_answer=lambda d: pd.to_datetime(d['Timestamp_answer'], format='mixed'))
    df_demographic = _read_table("df_demographic")

    df_merged = merge_demographics(df, df_demographic)
    df_merged = df_merged.dropna(subset=['Timestamp_answer']).copy()

    if group_by not in df_merged.columns:
        raise ValueError(f"group_by must be 'sex' or 'age_range', got '{group_by}'")

    n_slots_per_day = 24 // BIN_HOURS
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    full_grid = pd.DataFrame([
        {'day_of_week': d, 'hour_bin': b * BIN_HOURS}
        for d in range(7) for b in range(n_slots_per_day)
    ])
    full_grid['x'] = full_grid['day_of_week'] * n_slots_per_day + full_grid['hour_bin'] // BIN_HOURS

    COLORS  = plotly.colors.qualitative.D3
    groups  = sorted(df_merged[group_by].dropna().unique().tolist(), key=str)

    fig = go.Figure()

    for i, g in enumerate(groups):
        df_g = df_merged[df_merged[group_by].astype(str) == str(g)]
        if df_g.empty:
            continue
        n_indiv        = df_g['participant'].nunique()
        n_interactions = len(df_g)

        with contextlib.redirect_stdout(io.StringIO()):
            dist = compute_p_distribution(df_g.copy(), BIN_HOURS)

        merged = full_grid.merge(
            dist[['day_of_week', 'hour_bin', 'probability']],
            on=['day_of_week', 'hour_bin'], how='left'
        ).fillna(0)

        hover = merged.apply(
            lambda r: (
                f"<b>{day_names[int(r['day_of_week'])]}</b> "
                f"{int(r['hour_bin']):02d}:00–{int(r['hour_bin']) + BIN_HOURS:02d}:00"
                f"<br>P = {r['probability']:.4f}"
            ), axis=1
        )

        fig.add_trace(go.Scatter(
            x=merged['x'],
            y=merged['probability'],
            mode='lines',
            name=f"{g}  ({n_indiv} indiv., {n_interactions} interactions)",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertext=hover,
            hoverinfo='text',
        ))

    shapes = [
        dict(type='line',
             x0=d * n_slots_per_day - 0.5, x1=d * n_slots_per_day - 0.5,
             y0=0, y1=1, yref='paper',
             line=dict(color='#cccccc', width=1, dash='dash'))
        for d in range(1, 7)
    ]
    annotations = [
        dict(x=d * n_slots_per_day + n_slots_per_day / 2 - 0.5,
             y=1.06, xref='x', yref='paper',
             text=f'<b>{day_names[d]}</b>',
             showarrow=False, font=dict(size=12))
        for d in range(7)
    ]
    tick_vals = [d * n_slots_per_day + b for d in range(7) for b in range(n_slots_per_day) if b % 2 == 0]
    tick_text = [f"{b * BIN_HOURS:02d}h" for _ in range(7) for b in range(n_slots_per_day) if b % 2 == 0]

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45,
                   range=[-0.5, 7 * n_slots_per_day - 0.5]),
        yaxis=dict(title='Probability', gridcolor='#eeeeee'),
        title=dict(text=f'Interaction Probability Distribution by {group_by}  (bin={BIN_HOURS}h)',
                   font=dict(size=14)),
        legend=dict(title=group_by, font=dict(size=11)),
        height=480,
        margin=dict(t=90),
        plot_bgcolor='white',
        hovermode='x unified',
    )

    return fig


# ── plot_all_distribution (adapted from process_data.ipynb) ───────────────────

def plot_all_distribution(df_prob, df_pers_prob, df_final, wn, BIN_HOURS, output=None, n_pop_indiv=None, n_cohort_indiv=None, cohort_demographics=None):
    import plotly.graph_objects as go

    n_slots_per_day = 24 // BIN_HOURS
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    nb_pop     = int(df_prob['count'].sum())
    nb_pers    = int(df_pers_prob['count'].sum())
    nb_cohort  = int(df_final['count'].sum()) if 'count' in df_final.columns else 0

    full_grid = pd.DataFrame([
        {'day_of_week': d, 'hour_bin': b * BIN_HOURS}
        for d in range(7) for b in range(n_slots_per_day)
    ])
    full_grid['x'] = full_grid['day_of_week'] * n_slots_per_day + full_grid['hour_bin'] // BIN_HOURS

    pop  = full_grid.merge(df_prob[['day_of_week', 'hour_bin', 'probability']], on=['day_of_week', 'hour_bin'], how='left').fillna(0)
    pers = full_grid.merge(df_pers_prob[['day_of_week', 'hour_bin', 'probability']], on=['day_of_week', 'hour_bin'], how='left').fillna(0)
    fin  = full_grid.merge(df_final[['day_of_week', 'hour_bin', 'p_final']], on=['day_of_week', 'hour_bin'], how='left').fillna(0)

    def make_hover(df_row, col, label):
        return df_row.apply(
            lambda r: (
                f"<b>{label}</b><br>"
                f"{day_names[int(r['day_of_week'])]} "
                f"{int(r['hour_bin']):02d}:00–{int(r['hour_bin']) + BIN_HOURS:02d}:00"
                f"<br>P = {r[col]:.4f}"
            ), axis=1
        )

    if n_pop_indiv is not None:
        pop_label = f'Population ({n_pop_indiv} indiv., {nb_pop} interactions)'
    else:
        pop_label = f'Population ({nb_pop} interactions)'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pop['x'], y=pop['probability'],
        mode='lines', name=pop_label,
        line=dict(color='#4C72B0', width=1.5),
        hovertext=make_hover(pop, 'probability', 'Population'), hoverinfo='text',
    ))
    fig.add_trace(go.Scatter(
        x=pers['x'], y=pers['probability'],
        mode='lines', name=f'Patient ({nb_pers} interactions)',
        line=dict(color='#DD8452', width=1.5),
        hovertext=make_hover(pers, 'probability', 'Patient'), hoverinfo='text',
    ))
    cohort_desc = f'Tailored cohort ({n_cohort_indiv or "?"} indiv., {nb_cohort} interactions)'
    if cohort_demographics:
        demo_parts = []
        if cohort_demographics.get("sex"):
            demo_parts.append(f'sex={cohort_demographics["sex"]}')
        if cohort_demographics.get("age_range"):
            demo_parts.append(f'age={cohort_demographics["age_range"][0]}-{cohort_demographics["age_range"][1]}')
        if demo_parts:
            cohort_desc += f' [{", ".join(demo_parts)}]'

    fig.add_trace(go.Scatter(
        x=fin['x'], y=fin['p_final'],
        mode='lines', name=cohort_desc,
        line=dict(color='#2CA02C', width=2, dash='dash'),
        hovertext=make_hover(fin, 'p_final', 'Tailored cohort'), hoverinfo='text',
    ))

    # Red dots for best interaction times (output of get_top_k_peaks)
    if output is not None and not output.empty:
        peak_x, peak_y, peak_hover = [], [], []
        for _, row in output.iterrows():
            ts       = row['timestamp']
            dow      = ts.weekday()
            hour_bin = (ts.hour // BIN_HOURS) * BIN_HOURS
            x        = dow * n_slots_per_day + hour_bin // BIN_HOURS
            peak_x.append(x)
            peak_y.append(float(row['p_final']))
            peak_hover.append(
                f"<b>Best time #{len(peak_x)}</b><br>"
                f"{day_names[dow]} {ts.strftime('%H:%M')}<br>"
                f"p_final = {float(row['p_final']):.4f}"
            )
        fig.add_trace(go.Scatter(
            x=peak_x, y=peak_y,
            mode='markers',
            name='Best times',
            marker=dict(color='red', size=10, symbol='circle',
                        line=dict(color='darkred', width=1.5)),
            hovertext=peak_hover,
            hoverinfo='text',
        ))

    shapes = [
        dict(type='line',
             x0=d * n_slots_per_day - 0.5, x1=d * n_slots_per_day - 0.5,
             y0=0, y1=1, yref='paper',
             line=dict(color='#cccccc', width=1, dash='dash'))
        for d in range(1, 7)
    ]
    annotations = [
        dict(x=d * n_slots_per_day + n_slots_per_day / 2 - 0.5,
             y=1.06, xref='x', yref='paper',
             text=f'<b>{day_names[d]}</b>',
             showarrow=False, font=dict(size=12))
        for d in range(7)
    ]
    tick_vals = [d * n_slots_per_day + b for d in range(7) for b in range(n_slots_per_day) if b % 2 == 0]
    tick_text = [f"{b * BIN_HOURS:02d}h" for _ in range(7) for b in range(n_slots_per_day) if b % 2 == 0]

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45,
                   range=[-0.5, 7 * n_slots_per_day - 0.5]),
        yaxis=dict(title='Probability', gridcolor='#eeeeee'),
        title=dict(text=f'Probability Distributions — Population vs Patient vs Tailored Cohort  (bin={BIN_HOURS}h)',
                   font=dict(size=14)),
        legend=dict(font=dict(size=11)),
        height=480, margin=dict(t=90),
        plot_bgcolor='white', hovermode='x unified',
    )
    return fig


# ── MCP Tools ─────────────────────────────────────────────────────────────────

def patient_metadata(participant_id: str, age: str = "", sex: str = "") -> str:
    """
    Tool: patient_metadata

    Append a participant's demographic data to df_demographic.csv.
    Automatically creates new columns for any unknown demographic keys.

    Inputs:
      - participant_id : participant identifier string
      - age            : (optional) numeric age e.g. "45"
      - sex            : (optional) "male" or "female"

    Output:
      JSON confirming the added row and any new columns created.
    """
    try:
        if not participant_id.strip():
            return json.dumps({"error": "participant_id is required"})

        demo_data = {
            "age":            float(age.strip()) if age.strip() else None,
            "participantSex": sex.strip() if sex.strip() else None,
        }

        df = _read_table("df_demographic")

        # Detect and add new columns (existing rows get NaN)
        new_cols = [k for k in demo_data if k not in df.columns and k != "participant"]
        for col in new_cols:
            df[col] = pd.NA

        mask = df['participant'].astype(str) == participant_id.strip()
        if mask.any():
            # Update existing row
            with ENGINE.begin() as conn:
                conn.execute(text(
                    'UPDATE df_demographic SET age = :age, participantsex = :sex WHERE participant = :pid'
                ), {"age": demo_data["age"], "sex": demo_data["participantSex"], "pid": participant_id.strip()})
        else:
            # Insert new row
            _write_table(
                pd.DataFrame([{"participant": participant_id.strip(), **demo_data}]),
                "df_demographic", if_exists="append", index=False
            )

        return json.dumps({
            "status":              "updated" if mask.any() else "created",
            "participant":         participant_id.strip(),
            "demographics":        demo_data,
            "new_columns_created": new_cols,
        })
    except ValueError:
        return json.dumps({"error": "age must be a number"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_patient_scores(patient_id: str) -> dict:
    """MAE scores in minutes for each distribution, from best_times.csv rows with actual_time filled."""
    DECAY_ALPHA = 0.85

    def _compute_score(predicted: pd.Series, actual: pd.Series) -> float:
        diff_min = (predicted - actual).abs().dt.total_seconds() / 60
        n = len(diff_min)
        weights = pd.Series([DECAY_ALPHA ** (n - 1 - i) for i in range(n)])
        return round((diff_min.values * weights.values).sum() / weights.sum(), 1)

    try:
        df = _read_table("best_times")
        df.columns = df.columns.str.strip()
        df = df[df['patient_id'].astype(str) == str(patient_id)]
        df = df.sort_values("interaction_ID").reset_index(drop=True)
        df = df[df['actual_time'].notna()]
        if df.empty:
            return {"score_population": None, "score_patient": None, "score_tailored": None, "n_scored": 0}
        for col in ["best_time_population", "best_time_patient", "best_time_tailored", "actual_time"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        result = {"n_scored": len(df)}
        for score_key, col in [
            ("score_population", "best_time_population"),
            ("score_patient",    "best_time_patient"),
            ("score_tailored",   "best_time_tailored"),
        ]:
            valid = df[[col, "actual_time"]].dropna()
            result[score_key] = _compute_score(valid[col], valid["actual_time"]) if not valid.empty else None
        return result
    except Exception:
        return {"score_population": None, "score_patient": None, "score_tailored": None, "n_scored": 0}


def _select_best_distribution(scores, output_pop, output_pers, output_tailored):
    """
    Returns (winning_output, winner_label, rationale).
    Picks distribution with lowest MAE. Falls back to tailored if no scores exist.
    """
    sp = scores.get("score_population")
    ss = scores.get("score_patient")
    st = scores.get("score_tailored")
    n  = scores.get("n_scored", 0)

    dist_map = {
        "population":   (sp, output_pop),
        "personalized": (ss, output_pers),
        "tailored":     (st, output_tailored),
    }

    candidates = [
        (score, label, df)
        for label, (score, df) in dist_map.items()
        if score is not None and not df.empty
    ]

    if candidates:
        best_score, winner_label, winner_output = min(candidates, key=lambda x: x[0])
        scores_str = ", ".join(
            f"{lbl} ±{v:.0f} min" if v is not None else f"{lbl} N/A"
            for lbl, (v, _) in dist_map.items()
        )
        rationale = (
            f"Selected '{winner_label}' distribution (lowest historical error: ±{best_score:.0f} min). "
            f"Scores over {n} past interactions — {scores_str}."
        )
    else:
        winner_label  = "population"
        winner_output = output_pop if not output_pop.empty else output_tailored
        if n == 0:
            rationale = "No historical accuracy data yet. Returning population distribution."
        else:
            rationale = "Insufficient scored data for distribution selection. Returning population distribution."

    return winner_output, winner_label, rationale


def get_best_times(
    patient_id: str,
    target_date: str = "",
    k: int = 1,
    BIN_HOURS: int = 1,
    cutoff_date: str = "",
) -> str:
    """
    Tool: get_best_times

    Find the top K best times to interact with a patient by blending their
    personal interaction history with a demographically matched population.
    Demographics (age, sex) are looked up automatically from df_demographic.csv
    using the patient_id.

    Inputs:
      - patient_id  : patient number
      - target_date : criticality window — natural language extracted from user prompt
                      e.g. "tomorrow", "next monday", "in the next 3 days", "in the next 6 hours",
                      "45 minutes", "1 hour". If <= 1 hour, best time is now.
      - k           : number of time slots to return (default 3)
      - BIN_HOURS   : time bin size in hours (default 1)

    Output:
      JSON list of top K timestamps with their blended probability (p_final).
    """
    try:
        # Criticality check: if target_date specifies <= 1 hour, interact now
        _CRIT_RE = re.compile(r'(?:(\d+(?:\.\d+)?)\s*)?(hour|hours|h|min|mins|minute|minutes|m)\b', re.IGNORECASE)
        crit_match = _CRIT_RE.search((target_date or "").strip())
        if crit_match:
            val = float(crit_match.group(1)) if crit_match.group(1) else 1.0
            unit = crit_match.group(2).lower()
            crit_hours = val if unit.startswith('h') else val / 60
            if crit_hours <= 1:
                now = datetime.now()
                return json.dumps({
                    "patient_id": patient_id,
                    "target_date": target_date,
                    "selected_distribution": "critical",
                    "best_times": [{"timestamp": (now + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M"), "weekday": ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][(now + timedelta(minutes=2)).weekday()], "p_final": 1.0}],
                    "rationale": f"Criticality ({target_date}) is ≤ 1 hour. Best time to interact is now.",
                }, ensure_ascii=False)

        # Load data from database and split by patient_id
        df_all         = _read_table("interactions").assign(Timestamp_answer=lambda d: pd.to_datetime(d['Timestamp_answer'], format='mixed'))
        df_pers        = df_all[df_all['participant'].astype(str) == patient_id.strip()]
        df             = df_all[df_all['participant'].astype(str) != patient_id.strip()]
        df_demographic = _read_table("df_demographic")

        # Simulation / backtesting: restrict personalized history to before cutoff_date
        #if cutoff_date.strip():
           # df_pers = df_pers[df_pers['Timestamp_answer'] < pd.Timestamp(cutoff_date)]

        # Look up demographics from df_demographic.csv using patient_id
        patient_row = df_demographic[df_demographic['participant'] == patient_id.strip()]
        if not patient_row.empty:
            age_raw    = patient_row['age'].iloc[0] if 'age' in patient_row.columns else None
            sex_raw    = patient_row['participantSex'].iloc[0] if 'participantSex' in patient_row.columns else None
            age_val    = float(age_raw) if pd.notna(age_raw) else None
            sex_filter = {'female': 'f', 'male': 'm'}.get(sex_raw) if pd.notna(sex_raw) else None
        else:
            age_val, sex_filter = None, None

        # Select demographically matched population
        df_merged = merge_demographics(df, df_demographic)

        # Resolve age → age_range label for reporting
        age_bins, age_labels = [0, 40, 60, 120], ['<40', '40-60', '60+']
        age_range_used = None
        if age_val is not None:
            for i in range(len(age_bins) - 1):
                if age_bins[i] <= age_val < age_bins[i + 1]:
                    age_range_used = age_labels[i]
                    break

        # Population distribution uses ALL other patients (no demographic filter)
        # Compute probability distributions
        BIN_HOURS = int(BIN_HOURS) if BIN_HOURS is not None else 1
        with contextlib.redirect_stdout(io.StringIO()):
            df_prob = compute_p_distribution(df_merged.copy(), BIN_HOURS)
            if not df_pers.empty:
                df_pers_prob = compute_p_distribution(df_pers.copy(), BIN_HOURS)
            else:
                df_pers_prob = pd.DataFrame(columns=['day_of_week', 'hour_bin', 'count', 'probability'])

        # Tailored cohort selection
        cohort = select_tailored_cohort(df_demographic, patient_id=patient_id.strip())
        cohort_ids = df_demographic.loc[cohort.neighbor_indices, "participant"].tolist()
        df_cohort = df[df["participant"].isin(cohort_ids)]

        with contextlib.redirect_stdout(io.StringIO()):
            df_cohort_prob = compute_p_distribution(df_cohort.copy(), BIN_HOURS)

        start_day, end_day, start_time, end_time = parse_target_date(target_date)

        # Get top-K peaks for each distribution
        df_prob_final      = df_prob.rename(columns={'probability': 'p_final'})
        df_pers_prob_final = df_pers_prob.rename(columns={'probability': 'p_final'})
        df_cohort_prob_final = df_cohort_prob.rename(columns={'probability': 'p_final'})

        output_pop    = get_top_k_peaks(df_prob_final,        start_day, end_day, start_time, end_time, K=int(k))
        output_pers   = get_top_k_peaks(df_pers_prob_final,   start_day, end_day, start_time, end_time, K=int(k))
        output_cohort = get_top_k_peaks(df_cohort_prob_final, start_day, end_day, start_time, end_time, K=int(k))

        # The cohort distribution is the primary result
        output = output_cohort

        if output.empty:
            return json.dumps({"patient_id": patient_id, "best_times": []})

        # Describe the population distribution used
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        def fmt_rows(df, prob_col):
            return [
                {
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "weekday":   day_names[row["timestamp"].weekday()],
                    prob_col:    round(float(row["p_final"]), 6),
                }
                for _, row in df.iterrows()
            ]

        # Log to best_times table
        df_log = _read_table("best_times")
        interaction_id = len(df_log) + 1
        _write_table(pd.DataFrame([{
            "patient_id":           patient_id,
            "interaction_ID":       interaction_id,
            "target_date":          target_date,
            "best_time_population": output_pop['timestamp'].iloc[0].strftime("%Y-%m-%d %H:%M") if not output_pop.empty else None,
            "best_time_patient":    output_pers['timestamp'].iloc[0].strftime("%Y-%m-%d %H:%M") if not output_pers.empty else None,
            "best_time_tailored":   output['timestamp'].iloc[0].strftime("%Y-%m-%d %H:%M") if not output.empty else None,
            "actual_time":          None,
        }]), "best_times", if_exists="append", index=False)

        scores = _get_patient_scores(patient_id)
        winner_output, winner_label, rationale = _select_best_distribution(
            scores, output_pop, output_pers, output_cohort
        )

        return json.dumps({
            "patient_id":                   patient_id,
            "target_date":                  target_date,
            "cohort": {
                "pool_before":                cohort.pool_before,
                "active_categorical_values":  cohort.active_categorical_values,
                "pool_after_categorical":     cohort.pool_after_categorical,
                "pool_after_knn":             cohort.pool_after_knn,
                "age_range":                  list(cohort.age_range) if cohort.age_range else None,
                "fallback":                   cohort.fallback,
            },
            "selected_distribution":        winner_label,
            "best_times":                   fmt_rows(winner_output, "p_final"),
            "scores":    scores,
            "rationale": rationale,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


def add_interaction(participant_id: str, timestamps: str) -> str:
    """
    Tool: add_interaction

    Append one or more interaction rows to the interactions table.

    Inputs:
      - participant_id : patient identifier (e.g. "19")
      - timestamps     : one or more interaction datetimes, comma-separated
                         (e.g. "2024-03-01 14:30:00" or "2024-03-01 14:30, 2024-03-02 09:00, 2024-03-03 18:15")

    Output:
      JSON confirming the added rows, or an error message.
    """
    try:
        if not participant_id.strip():
            return json.dumps({"error": "participant_id is required"})

        raw_list = [t.strip() for t in (timestamps or "").split(",") if t.strip()]
        if not raw_list:
            return json.dumps({"error": "at least one timestamp is required"})

        added = []
        for ts_raw in raw_list:
            ts = pd.to_datetime(ts_raw)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

            _write_table(pd.DataFrame([{
                "participant":      participant_id.strip(),
                "Timestamp_answer": ts_str,
            }]), "interactions", if_exists="append", index=False)

            added.append(ts_str)

        # Fill actual_time in unmatched best_times rows for this patient
        df_log = _read_table("best_times")
        df_log.columns = df_log.columns.str.strip()
        mask = (
            (df_log['patient_id'].astype(str) == participant_id.strip()) &
            (df_log['actual_time'].isna() | (df_log['actual_time'].astype(str).str.strip() == ''))
        )
        unmatched_indices = df_log[mask].index.tolist()
        for i, ts_str in enumerate(added):
            if i >= len(unmatched_indices):
                break
            row = df_log.loc[unmatched_indices[i]]
            with ENGINE.begin() as conn:
                conn.execute(text(
                    'UPDATE best_times SET actual_time = :ts WHERE patient_id = :pid AND interaction_id = :iid'
                ), {"ts": ts_str, "pid": participant_id.strip(), "iid": int(row["interaction_ID"])})

        return json.dumps({
            "status":            "ok",
            "participant":       participant_id.strip(),
            "timestamps_added":  added,
            "count":             len(added),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def add_intervention_context(intervention: str, context: str) -> str:
    """
    Tool: add_intervention_context

    Add or update an intervention entry in the intervention_context table.
    If the intervention name already exists its context is overwritten.

    Inputs:
      - intervention : name of the intervention (e.g. "CanRelax")
      - context      : free-text description of the intervention context
    """
    try:
        intervention = intervention.strip()
        context      = context.strip()
        if not intervention:
            return json.dumps({"error": "intervention name is required"})
        if not context:
            return json.dumps({"error": "context is required"})

        df_ctx = _read_table("intervention_context")
        df_ctx.columns = df_ctx.columns.str.strip()

        if intervention in df_ctx["intervention"].values:
            with ENGINE.begin() as conn:
                conn.execute(text(
                    "UPDATE intervention_context SET context = :ctx WHERE intervention = :name"
                ), {"ctx": context, "name": intervention})
            status = "updated"
        else:
            _write_table(
                pd.DataFrame([{"intervention": intervention, "context": context}]),
                "intervention_context", if_exists="append", index=False
            )
            status = "added"
        return json.dumps({"status": status, "intervention": intervention, "context": context})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Gradio UI + MCP exposure ──────────────────────────────────────────────────

with gr.Blocks() as demo:

        # ── patient_metadata ──
    gr.Markdown("### Patient metadata")
    with gr.Row():
        meta_pid_in  = gr.Textbox(label="participant_id", placeholder="e.g. 19")
        meta_age_in  = gr.Textbox(label="age (optional)", placeholder="e.g. 45")
        meta_sex_in  = gr.Dropdown(label="sex (optional)", choices=["", "male", "female"], value="")
    btn_meta = gr.Button("Add metadata")
    out_meta = gr.Textbox(label="Result (JSON)", lines=3)
    btn_meta.click(
        fn=patient_metadata,
        inputs=[meta_pid_in, meta_age_in, meta_sex_in],
        outputs=out_meta,
        api_name="patient_metadata",
    )

        # ── add_intervention_context ──
    gr.Markdown("### Intervention Metadata")
    with gr.Row():
        ctx_name_in = gr.Textbox(label="intervention", placeholder="e.g. CanRelax")
        ctx_text_in = gr.Textbox(label="context", placeholder="Free-text description of the intervention", lines=3)
    btn_ctx = gr.Button("Add / update intervention context")
    out_ctx = gr.Textbox(label="Result (JSON)", lines=3)
    btn_ctx.click(
        fn=add_intervention_context,
        inputs=[ctx_name_in, ctx_text_in],
        outputs=out_ctx,
        api_name="add_intervention_context",
    )
    
    gr.Markdown("## Receptivity MCP Server")

    debug_toggle = gr.Checkbox(label="Debug mode", value=False)

    with gr.Row():
        patient_in    = gr.Textbox(label="patient_id",    placeholder="e.g. 19")
        date_in       = gr.Textbox(label="Criticality",   placeholder="e.g. in the next 6 hours / 45 minutes / tomorrow")
        k_in          = gr.Number(label="k",              value=1, precision=0)
        bin_in        = gr.Number(label="BIN_HOURS",      value=1, precision=0)
    cutoff_in = gr.Textbox(label="cutoff_date (debug — simulation only)",
                           placeholder="e.g. 2024-06-01", visible=False)
    debug_toggle.change(fn=lambda v: gr.update(visible=v), inputs=debug_toggle, outputs=cutoff_in)

    with gr.Row():
        btn     = gr.Button("Get best times")
        btn_plt = gr.Button("Plot all distributions")

    out      = gr.Textbox(label="Result (JSON)", lines=12)
    plot_all = gr.Plot(label="Population vs Patient vs Final")

    btn.click(
        fn=get_best_times,
        inputs=[patient_in, date_in, k_in, bin_in, cutoff_in],
        outputs=out,
        api_name="get_best_times",
    )


        # ── add_interaction ──
    gr.Markdown("### Add interaction")
    with gr.Row():
        add_pid_in = gr.Textbox(label="participant_id", placeholder="e.g. 62d967a58b87bc0009032e53")
        add_ts_in  = gr.Textbox(label="timestamps (comma-separated)", placeholder="e.g. 2024-03-01 14:30, 2024-03-02 09:00")
    btn_add  = gr.Button("Add interaction")
    out_add  = gr.Textbox(label="Result (JSON)", lines=3)
    btn_add.click(
        fn=add_interaction,
        inputs=[add_pid_in, add_ts_in],
        outputs=out_add,
        api_name="add_interaction",
    )


    def _plot_all_distributions(patient_id, target_date, k, BIN_HOURS):
        BIN_HOURS = int(BIN_HOURS) if BIN_HOURS is not None else 1
        try:
            df_all         = _read_table("interactions").assign(Timestamp_answer=lambda d: pd.to_datetime(d['Timestamp_answer'], format='mixed'))
            df_pers        = df_all[df_all['participant'].astype(str) == str(patient_id).strip()]
            df             = df_all[df_all['participant'].astype(str) != str(patient_id).strip()]
            df_demographic = _read_table("df_demographic")

            patient_row = df_demographic[df_demographic['participant'] == str(patient_id).strip()]
            if not patient_row.empty:
                age_raw    = patient_row['age'].iloc[0] if 'age' in patient_row.columns else None
                sex_raw    = patient_row['participantSex'].iloc[0] if 'participantSex' in patient_row.columns else None
                age_val    = float(age_raw) if pd.notna(age_raw) else None
                sex_filter = {'female': 'f', 'male': 'm'}.get(sex_raw) if pd.notna(sex_raw) else None
            else:
                age_val, sex_filter = None, None

            df_merged = merge_demographics(df, df_demographic)

            with contextlib.redirect_stdout(io.StringIO()):
                df_prob = compute_p_distribution(df_merged.copy(), BIN_HOURS)
                if not df_pers.empty:
                    df_pers_prob = compute_p_distribution(df_pers.copy(), BIN_HOURS)
                else:
                    df_pers_prob = pd.DataFrame(columns=['day_of_week', 'hour_bin', 'count', 'probability'])

            # Tailored cohort distribution as "final"
            cohort = select_tailored_cohort(df_demographic, patient_id=str(patient_id).strip())
            cohort_ids = df_demographic.loc[cohort.neighbor_indices, "participant"].tolist()
            df_cohort = df[df["participant"].isin(cohort_ids)]
            with contextlib.redirect_stdout(io.StringIO()):
                df_cohort_prob = compute_p_distribution(df_cohort.copy(), BIN_HOURS)

            start_day, end_day, start_time, end_time = parse_target_date(target_date or None)
            output = get_top_k_peaks(df_cohort_prob.rename(columns={'probability': 'p_final'}),
                                     start_day, end_day, start_time, end_time, K=int(k))

            df_cohort_final = df_cohort_prob.rename(columns={'probability': 'p_final'})
            cohort_demo = {
                "sex": cohort.active_categorical_values.get("participantSex"),
                "age_range": cohort.age_range,
            }
            return plot_all_distribution(df_prob, df_pers_prob, df_cohort_final, 0.0, BIN_HOURS,
                                         output=output, n_pop_indiv=df_merged['participant'].nunique(),
                                         n_cohort_indiv=df_cohort['participant'].nunique(),
                                         cohort_demographics=cohort_demo)
        except Exception as e:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(); ax.set_title(f"Error: {e}"); ax.axis("off"); return fig

    btn_plt.click(
        fn=_plot_all_distributions,
        inputs=[patient_in, date_in, k_in, bin_in],
        outputs=plot_all,
        api_name="plot_all_distributions",
    )

    


    # ── plot_p_distribution_demographic ──
    gr.Markdown("### Population distribution by demographic group")
    with gr.Row():
        group_by_in = gr.Dropdown(label="group_by", choices=["sex", "age_range"], value="sex")
        bin_demo_in = gr.Number(label="BIN_HOURS", value=4, precision=0)
    btn_demo = gr.Button("Plot demographic distribution")
    plot_demo = gr.Plot(label="Distribution by group")
    btn_demo.click(
        fn=plot_p_distribution_demographic,
        inputs=[group_by_in, bin_demo_in],
        outputs=plot_demo,
        api_name="plot_p_distribution_demographic",
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, server_name="0.0.0.0", server_port=7860)
