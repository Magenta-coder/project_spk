from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'secret123'

# Fungsi AHP: menghitung bobot dan mengecek konsistensi
def calculate_ahp_weights(matrix):
    matrix = np.array(matrix)
    n = matrix.shape[0]

    # Langkah 1: Normalisasi matriks dan hitung bobot
    col_sum = matrix.sum(axis=0)
    norm_matrix = matrix / col_sum
    weights = norm_matrix.mean(axis=1)

    # Langkah 2: Cek konsistensi
    weighted_sum = np.dot(matrix, weights)
    lamda_max = (weighted_sum / weights).mean()
    CI = (lamda_max - n) / (n - 1)
    RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
               7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI

    if CR > 0.1:
        raise ValueError(f'Consistency Ratio too high: {CR:.2f}. Please revise pairwise matrix.')

    return weights

# Fungsi untuk menghitung bobot GAP diskret
def get_gap_weight(gap):
    gap_weights = {
        0: 5, 1: 4.5, -1: 4.5,
        2: 4, -2: 4,
        3: 3.5, -3: 3.5,
        4: 3, -4: 3,
        5: 2.5, -5: 2.5
    }
    return gap_weights.get(int(gap), 1)

# Fungsi Profile Matching sederhana
def profile_matching(alternatives_df, profiles_series, weights=None):
    gap_scores = []
    for _, alt in alternatives_df.iterrows():
        gaps = alt - profiles_series
        gap_weights = gaps.apply(get_gap_weight)
        score = gap_weights.mean()  # Rata-rata bobot GAP
        gap_scores.append(score)
    return pd.Series(gap_scores, index=alternatives_df.index)

# Fungsi TOPSIS
def topsis_ranking(df, weights, impacts):
    matrix = df.values.astype(float)
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    v = norm * weights
    ideal_best = np.array([v[:,j].max() if imp == '+' else v[:,j].min()
                           for j, imp in enumerate(impacts)])
    ideal_worst = np.array([v[:,j].min() if imp == '+' else v[:,j].max()
                            for j, imp in enumerate(impacts)])
    dist_best = np.sqrt(((v - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((v - ideal_worst)**2).sum(axis=1))
    ci = dist_worst / (dist_best + dist_worst)
    return ci

# ROUTING
@app.route('/')
def home():
    return redirect(url_for('kriteria'))

@app.route('/kriteria', methods=['GET', 'POST'])
def kriteria():
    if request.method == 'POST':
        session['criteria'] = request.form.getlist('criteria[]')
        session['impact']   = request.form.getlist('impact[]')       # e.g. ['+','-','+']
        session['profile']  = list(map(float, request.form.getlist('profile[]')))
        return redirect(url_for('pairwise'))
    return render_template('kriteria.html')

@app.route('/pairwise', methods=['GET', 'POST'])
def pairwise():
    criteria = session.get('criteria', [])
    n = len(criteria)
    if n == 0:
        return redirect(url_for('kriteria'))
    if request.method == 'POST':
        matrix = []
        for i in range(n):
            row = [float(request.form.get(f'pairwise_{i}_{j}', 1))
                   for j in range(n)]
            matrix.append(row)
        session['pairwise'] = matrix
        return redirect(url_for('alternatif'))
    return render_template('pairwise.html', criteria=criteria)

@app.route('/alternatif', methods=['GET', 'POST'])
def alternatif():
    criteria = session.get('criteria', [])
    if request.method == 'POST':
        alt_names = request.form.getlist('alt_name[]')
        alt_data  = [list(map(float, request.form.getlist(f'alt_{i}[]')))
                     for i in range(len(alt_names))]
        session['alt_names'] = alt_names
        session['alt_data']  = alt_data
        return redirect(url_for('hasil'))
    return render_template('alternatif.html', criteria=criteria)

@app.route('/hasil')
def hasil():
    crit    = session['criteria']
    impacts = session['impact']
    profiles= np.array(session['profile'])
    pairM   = np.array(session['pairwise'])
    alt_names = session['alt_names']
    alt_data  = session['alt_data']

    try:
        weights = calculate_ahp_weights(pairM)
    except ValueError as e:
        return render_template('error.html', error_message=str(e))

    df_alt = pd.DataFrame(alt_data, index=alt_names, columns=crit)
    profiles_series = pd.Series(profiles, index=crit)

    # PM Score
    pm_scores = profile_matching(df_alt, profiles_series)

    # TOPSIS
    topsis_ci = topsis_ranking(df_alt, weights, impacts)

    # Gabungkan hasil
    results = pd.DataFrame({
        'PM Score': pm_scores.round(4),
        'Topsis CI': topsis_ci.round(4)
    })
    results['Topsis Rank'] = results['Topsis CI'].rank(ascending=False).astype(int)
    results = results.sort_values('Topsis Rank')

    return render_template('hasil.html', tables=[results.to_html(classes='table table-bordered')])

if __name__ == '__main__':
    app.run(debug=True)