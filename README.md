# Student Performance Analysis & Pass/Fail Predictor — Streamlit App

A polished, tabbed dashboard built with Streamlit:

1. **🔮 Predictor** — enter a student's study habits and background to get a Pass/Fail prediction with a confidence score and a Low/Medium/High risk badge. Shows *why* the model made its call (local feature contributions for Logistic Regression, overall importance for tree models), plus a radar chart comparing the student to the class average. Includes a **Compare two students** mode for side-by-side what-if analysis.
2. **📋 Data Explorer** — filter by involvement, school type, result, study hours, and attendance; browse and download the filtered data; quick pass/fail and score-distribution charts.
3. **📈 Insights** — score breakdowns by study hours, attendance, parental involvement, and more, plus a correlation heatmap and a build-your-own scatter plot.
4. **🧪 Model Lab** — full model comparison table and chart, confusion matrices, ROC curves, and feature importance for Logistic Regression, Decision Tree, and Random Forest.
5. **📝 Findings** — auto-computed insights and a written conclusion that update live with the threshold.

A sidebar lets you adjust the **Pass/Fail threshold** and pick which model powers predictions (or leave it on **Auto**, which always uses the best-performing model) — every tab updates live when you change either.

## Project files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (5 tabs) |
| `student_cleaned.csv` | Cleaned dataset the app loads |
| `StudentPerformanceFactors_messy.csv` | Original raw dataset, kept for reference |
| `student_analysis.ipynb` | Notebook: cleans the raw data step by step |
| `student_visualization.ipynb` | Notebook: exploratory data analysis / charts |
| `Student_model.ipynb` | Notebook: trains & compares the 3 models, saves `student_pass_fail_model.pkl` |
| `student_pass_fail_model.pkl` | Saved best-model bundle (model + scaler + feature metadata) via `joblib` |
| `requirements.txt` | Python dependencies |

## How to run

1. Make sure `app.py` and `student_cleaned.csv` are in the **same folder**.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. It will open automatically at `http://localhost:8501`.

## Deploy for free — Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `student_cleaned.csv` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → pick your repo/branch → set **Main file path** to `app.py` → **Deploy**.
4. You'll get a public URL like `https://<your-app-name>.streamlit.app` within a minute or two.

## Notes

- The three models (Logistic Regression, Decision Tree, Random Forest) are trained **live**, right when the app starts (and retrained instantly whenever you move the threshold slider), using the same cleaning/encoding logic as the notebooks. Training is cached (`st.cache_resource`, keyed by threshold), so it only reruns when the threshold actually changes.
- `student_pass_fail_model.pkl` is produced by `Student_model.ipynb` as a standalone, reusable artifact (e.g. for a separate script or API) — the app doesn't depend on it, but you can load it with `joblib.load(...)` if you want to reuse the trained model elsewhere.
- The "why this prediction" explanation is a true local (per-student) breakdown for Logistic Regression, based on its coefficients; for the tree-based models it shows the model's overall feature importance instead, since a simple per-instance breakdown isn't available without an extra library like SHAP.

