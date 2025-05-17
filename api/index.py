from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'secret123'  # Wajib untuk session

# Fungsi AHP
def calculate_ahp_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(np.real(eigvals))
    weights = np.real(eigvecs[:, max_index])
    weights = weights / np.sum(weights)
    return weights

# Fungsi Profile Matching (gap-based similarity)
def profile_matching(alternatives_df, profiles_series, weights):
    # profiles_series: target nilai per kriteria
    # weights: bobot kriteria (AHP)
    scores = []
    for _, alt in alternatives_df.iterrows():
        gaps = alt - profiles_series
        max_gap = np.max(np.abs(gaps))
        if max_gap == 0:
            pref = np.ones_like(gaps)
        else:
            pref = 1 - (np.abs(gaps) / max_gap)
        scores.append(np.dot(pref, weights))
    return pd.Series(scores, index=alternatives_df.index)

# Fungsi TOPSIS
def topsis_ranking(df, weights, impacts):
    matrix = df.values.astype(float)
    # 1. Normalisasi
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    # 2. Bobot
    v = norm * weights
    # 3. Tentukan ideal best dan worst
    ideal_best = np.array([v[:,j].max() if imp=='+' else v[:,j].min()
                            for j,imp in enumerate(impacts)])
    ideal_worst = np.array([v[:,j].min() if imp=='+' else v[:,j].max()
                             for j,imp in enumerate(impacts)])
    # 4. Hitung jarak
    dist_best = np.sqrt(((v - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((v - ideal_worst)**2).sum(axis=1))
    # 5. Composite Index
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
        return redirect(url_for('kriteria'))  # ============== diperbaiki
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
    # Ambil data dari session
    crit    = session['criteria']
    impacts = session['impact']
    profiles= np.array(session['profile'])
    pairM   = np.array(session['pairwise'])
    alt_names = session['alt_names']
    alt_data  = session['alt_data']

    # Hitung bobot AHP
    weights = calculate_ahp_weights(pairM)

    # DataFrame alternatif
    df_alt = pd.DataFrame(alt_data, index=alt_names, columns=crit)
    profiles_series = pd.Series(profiles, index=crit)

    # Profile Matching
    pm_scores = profile_matching(df_alt, profiles_series, weights)

    # TOPSIS
    topsis_ci = topsis_ranking(df_alt, weights, impacts)

    # Gabungkan hasil
    results = pd.DataFrame({
        'PM Score': pm_scores,
        'Topsis CI': topsis_ci
    })
    results['Topsis Rank'] = results['Topsis CI'].rank(ascending=False).astype(int)
    results = results.sort_values('Topsis Rank')

    return render_template('hasil.html', tables=[results.to_html(classes='table table-bordered')])

# if __name__ == '__main__':
#     app.run(debug=True)
