from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# AHP
def calculate_ahp_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(np.real(eigvals))
    weights = np.real(eigvecs[:, max_index])
    weights = weights / np.sum(weights)
    return weights

# Profile Matching
def profile_matching(alternatives_df, profiles_series, criteria, weights):
    scores = []
    for _, alt in alternatives_df.iterrows():
        gaps = alt[criteria] - profiles_series
        pref = 1 - (np.abs(gaps) / np.max(np.abs(gaps)))
        score = np.dot(pref, weights)
        scores.append(score)
    return pd.Series(scores, index=alternatives_df.index)

# TOPSIS
def topsis_ranking(matrix, weights, impacts):
    norm = matrix / np.sqrt((matrix**2).sum())
    v = norm * weights
    ideal_best = np.where(impacts == '+', v.max(), v.min())
    ideal_worst = np.where(impacts == '+', v.min(), v.max())
    dist_best = np.sqrt(((v - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((v - ideal_worst)**2).sum(axis=1))
    ci = dist_worst / (dist_best + dist_worst)
    return ci

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        criteria = request.form.getlist('criteria[]')
        impacts = request.form.getlist('impact[]')
        profiles = list(map(float, request.form.getlist('profile[]')))
        alt_names = request.form.getlist('alt_name[]')
        num_criteria = len(criteria)

        # Ambil pairwise matrix
        matrix = []
        for i in range(num_criteria):
            row = []
            for j in range(num_criteria):
                val = float(request.form.get(f'pairwise_{i}_{j}', 1))
                row.append(val)
            matrix.append(row)
        crit_matrix = np.array(matrix)
        weights = calculate_ahp_weights(crit_matrix)

        # Ambil nilai alternatif
        alt_data = []
        for i in range(len(alt_names)):
            alt_values = list(map(float, request.form.getlist(f'alt_{i}[]')))
            alt_data.append(alt_values)
        df_alt = pd.DataFrame(alt_data, index=alt_names, columns=criteria)
        profiles_series = pd.Series(profiles, index=criteria)

        pm_scores = profile_matching(df_alt, profiles_series, criteria, weights)
        ci = topsis_ranking(df_alt, weights, np.array(impacts))

        results = pd.DataFrame({
            'PM Score': pm_scores,
            'Topsis CI': ci,
            'Topsis Rank': ci.rank(ascending=False)
        }, index=alt_names)

        return render_template('results.html', tables=[results.to_html(classes='table table-bordered')])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
