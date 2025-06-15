from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import io
from math import isfinite

app = Flask(__name__)
app.secret_key = 'secret123'


# --- Helper Functions ---

def calculate_ahp_weights(matrix):
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]

    if n < 2:
        return np.array([1.0]) if n == 1 else np.array([])

    col_sum = matrix.sum(axis=0)
    if (col_sum == 0).any():
        raise ValueError("One or more columns in pairwise matrix sum to zero.")

    norm_matrix = matrix / col_sum
    weights = norm_matrix.mean(axis=1)
    weights = weights / weights.sum()

    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum / weights)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12,
               6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0

    if CR > 0.1:
        raise ValueError(f'Consistency ratio too high: {CR:.2f}')

    return weights


def normalize_series(series, min_val, max_val, min_target=1, max_target=5, inverse=False):
    if inverse:
        return ((max_val - series) / (max_val - min_val)) * (max_target - min_target) + min_target
    else:
        return ((series - min_val) / (max_val - min_val)) * (max_target - min_target) + min_target


def normalize_all_criteria(df_alt, criteria, impacts, profiles, crit_types, min_ideals, max_ideals):
    df_normalized = df_alt.copy()
    profiles_normalized = pd.Series(index=criteria, dtype=float)

    for i, col in enumerate(criteria):
        current_impact = impacts[i]
        crit_type = crit_types[i]
        series = df_alt[col]

        if series.isna().all():
            df_normalized[col] = np.nan
            profiles_normalized[col] = np.nan
            continue

        min_val = series.min()
        max_val = series.max()

        if crit_type == 'scaled':
            if current_impact == '+':
                df_normalized[col] = normalize_series(series, min_val, max_val)
                if not pd.isna(profiles[i]):
                    profiles_normalized[col] = normalize_series(
                        pd.Series([profiles[i]]), min_val, max_val
                    ).iloc[0]
                else:
                    profiles_normalized[col] = 5.0
            else:
                df_normalized[col] = normalize_series(series, min_val, max_val, inverse=True)
                if not pd.isna(profiles[i]):
                    profiles_normalized[col] = normalize_series(
                        pd.Series([profiles[i]]), min_val, max_val, inverse=True
                    ).iloc[0]
                else:
                    profiles_normalized[col] = 5.0

        elif crit_type == 'range':
            m_ideal = min_ideals[i]
            x_ideal = max_ideals[i]

            def score_value(x):
                if pd.isna(x) or not isfinite(x):
                    return np.nan
                if m_ideal <= x <= x_ideal:
                    return 5.0
                elif x < m_ideal:
                    penalty = (m_ideal - x) / (x_ideal - m_ideal)
                    return max(1.0, 5.0 - penalty * 4)
                else:
                    penalty = (x - x_ideal) / (x_ideal - m_ideal)
                    return max(1.0, 5.0 - penalty * 4)

            df_normalized[col] = series.apply(score_value)
            profiles_normalized[col] = 5.0

    return df_normalized.fillna(0), profiles_normalized.fillna(0)


def get_gap_weight(gap):
    gap = round(gap)
    gap_weights = {
        0: 5,
        1: 4.5, -1: 4.5,
        2: 4, -2: 4,
        3: 3.5, -3: 3.5,
        4: 3, -4: 3,
        5: 2.5, -5: 2.5
    }
    return gap_weights.get(gap, 1)


def profile_matching(alternatives_df, profiles_series):
    gap_scores = []
    for _, alt in alternatives_df.iterrows():
        gaps = alt - profiles_series
        weights = gaps.apply(get_gap_weight)
        gap_scores.append(weights.mean())
    return pd.Series(gap_scores, index=alternatives_df.index)


def topsis_ranking(df, weights, impacts):
    matrix = df.values.astype(float)
    norm_divisor = np.sqrt((matrix ** 2).sum(axis=0))
    norm_matrix = matrix / np.where(norm_divisor == 0, np.finfo(float).eps, norm_divisor)
    weighted = norm_matrix * weights
    ideal_best = np.array([
        weighted[:, j].max() if imp == '+' else weighted[:, j].min()
        for j, imp in enumerate(impacts)
    ])
    ideal_worst = np.array([
        weighted[:, j].min() if imp == '+' else weighted[:, j].max()
        for j, imp in enumerate(impacts)
    ])
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    closeness = dist_worst / (dist_best + dist_worst + np.finfo(float).eps)
    return closeness, norm_matrix, weighted, ideal_best, ideal_worst


# --- Flask Routes ---

@app.route('/')
def home():
    return render_template('splash.html')


@app.route('/kriteria', methods=['GET', 'POST'])
def kriteria():
    if request.method == 'POST':
        criteria = []
        impacts = []
        profile_input_raw = []

        if 'file_upload' in request.files and request.files['file_upload'].filename != '':
            file = request.files['file_upload']
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
                else:
                    df = pd.read_excel(file)

                if not all(col in df.columns for col in ['Nama Kriteria', 'Dampak', 'Nilai Ideal']):
                    flash('File harus mengandung kolom: Nama Kriteria, Dampak, Nilai Ideal', 'danger')
                    return redirect(url_for('kriteria'))

                for _, row in df.iterrows():
                    criteria.append(str(row['Nama Kriteria']))
                    impacts.append(str(row['Dampak']))
                    profile_input_raw.append(str(row['Nilai Ideal']) if pd.notna(row['Nilai Ideal']) else '')

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(url_for('kriteria'))
        else:
            criteria = request.form.getlist('criteria[]')
            impacts = request.form.getlist('impact[]')
            profile_input_raw = request.form.getlist('profile[]')

            if not criteria or all(not c.strip() for c in criteria):
                flash('Harap masukkan setidaknya satu kriteria', 'danger')
                return redirect(url_for('kriteria'))

        profiles = []
        crit_types = []
        min_ideals = []
        max_ideals = []

        for i, val in enumerate(profile_input_raw):
            try:
                profile_val = float(val) if val.strip() else np.nan
            except:
                profile_val = np.nan

            if pd.isna(profile_val):
                profiles.append(np.nan)
                crit_types.append('scaled')
                min_ideals.append(np.nan)
                max_ideals.append(np.nan)
            elif profile_val > 5:
                crit_types.append('range')
                range_dev = profile_val * 0.1
                min_ideals.append(profile_val - range_dev)
                max_ideals.append(profile_val + range_dev)
                profiles.append(np.nan)
            else:
                crit_types.append('scaled')
                profiles.append(profile_val)
                min_ideals.append(np.nan)
                max_ideals.append(np.nan)

        session['criteria'] = criteria
        session['impact'] = impacts
        session['profile'] = profiles
        session['crit_types'] = crit_types
        session['min_ideals'] = min_ideals
        session['max_ideals'] = max_ideals

        return redirect(url_for('pairwise'))

    return render_template('kriteria.html')


@app.route('/pairwise', methods=['GET', 'POST'])
def pairwise():
    criteria = session.get('criteria', [])
    if not criteria:
        flash('Tidak ada kriteria yang ditemukan', 'warning')
        return redirect(url_for('kriteria'))

    if request.method == 'POST':
        n = len(criteria)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                val_str = request.form.get(f'pairwise_{i}_{j}')
                try:
                    val = float(val_str) if val_str and val_str.strip() else (1.0 if i == j else np.nan)
                except:
                    val = 1.0 if i == j else np.nan
                row.append(val)
            matrix.append(row)

        matrix_np = np.array(matrix, dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix_np[i, j] = 1.0
                elif np.isnan(matrix_np[i, j]) and not np.isnan(matrix_np[j, i]):
                    matrix_np[i, j] = 1 / matrix_np[j, i] if matrix_np[j, i] != 0 else np.nan

        if np.isnan(matrix_np).any():
            flash('Matriks berpasangan mengandung nilai tidak valid', 'danger')
            return render_template('pairwise.html', criteria=criteria)

        session['pairwise'] = matrix_np.tolist()
        return redirect(url_for('alternatif'))

    return render_template('pairwise.html', criteria=criteria)


@app.route('/alternatif', methods=['GET', 'POST'])
def alternatif():
    criteria = session.get('criteria', [])
    crit_types = session.get('crit_types', [])

    if not criteria:
        flash('Tidak ada kriteria yang ditemukan', 'warning')
        return redirect(url_for('kriteria'))

    if request.method == 'POST':
        alt_names = request.form.getlist('alt_name[]')
        if not alt_names or all(not name.strip() for name in alt_names):
            flash('Harap masukkan nama alternatif', 'danger')
            return render_template('alternatif.html', criteria=criteria, crit_types=crit_types)

        alt_data = []
        for i in range(len(alt_names)):
            row = []
            for k_idx in range(len(criteria)):
                crit_type = crit_types[k_idx]

                if crit_type == 'scaled':
                    val_str = request.form.get(f'alt_{i}_scaled_{k_idx}')
                    try:
                        val = float(val_str) if val_str and val_str.strip() else np.nan
                    except:
                        val = np.nan
                    row.append(val)
                else:
                    min_str = request.form.get(f'alt_{i}_range_min_{k_idx}')
                    max_str = request.form.get(f'alt_{i}_range_max_{k_idx}')
                    try:
                        min_val = float(min_str) if min_str and min_str.strip() else np.nan
                        max_val = float(max_str) if max_str and max_str.strip() else np.nan
                    except:
                        min_val = max_val = np.nan

                    if pd.isna(min_val) and pd.isna(max_val):
                        row.append(np.nan)
                    elif pd.isna(min_val):
                        row.append(max_val)
                    elif pd.isna(max_val):
                        row.append(min_val)
                    elif session['impact'][k_idx] == '+':
                        row.append(max(min_val, max_val))
                    else:
                        row.append(min(min_val, max_val))
            alt_data.append(row)

        session['alt_names'] = alt_names
        session['alt_data'] = alt_data
        return redirect(url_for('hasil'))

    return render_template('alternatif.html', criteria=criteria, crit_types=crit_types)


@app.route('/hasil')
def hasil():
    required_keys = ['criteria', 'impact', 'profile', 'crit_types',
                     'min_ideals', 'max_ideals', 'pairwise', 'alt_names', 'alt_data']
    if not all(key in session for key in required_keys):
        flash('Data sesi tidak lengkap', 'danger')
        return redirect(url_for('kriteria'))

    try:
        criteria = session['criteria']
        impacts = session['impact']
        profiles = np.array(session['profile'], dtype=float)
        crit_types = session['crit_types']
        min_ideals = np.array(session['min_ideals'], dtype=float)
        max_ideals = np.array(session['max_ideals'], dtype=float)
        pairwise = np.array(session['pairwise'], dtype=float)
        alt_names = session['alt_names']
        alt_data = np.array(session['alt_data'], dtype=float)

        weights = calculate_ahp_weights(pairwise)
        df_alt = pd.DataFrame(alt_data, index=alt_names, columns=criteria)
        df_norm, profiles_norm = normalize_all_criteria(
            df_alt, criteria, impacts, profiles, crit_types, min_ideals, max_ideals
        )
        pm_scores = profile_matching(df_norm, profiles_norm)
        topsis_scores, _, _, _, _ = topsis_ranking(df_norm, weights, impacts)

        results = pd.DataFrame({
            'PM Score': pm_scores.round(4),
            'TOPSIS Score': topsis_scores.round(4),
            'Total Score': (0.5 * pm_scores + 0.5 * topsis_scores).round(4)
        }, index=alt_names)

        results['Rank'] = results['Total Score'].rank(ascending=False, method='min').astype(int)
        results = results.sort_values('Rank')

        # Convert to HTML and split into header and rows
        results_html = results.to_html(classes='table table-bordered', index_names=False)

        return render_template('hasil.html',
                               tables=[results_html],
                               criteria=criteria)

    except Exception as e:
        return render_template('error.html', error_message=f"Error dalam perhitungan: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)