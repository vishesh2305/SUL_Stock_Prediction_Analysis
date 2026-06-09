# SUL Stock Prediction — Interactive Dashboard

A Streamlit companion to the project's Jupyter notebooks. The notebooks are
**never modified**; this dashboard re-applies their feature pipeline and
re-fits the same five classifiers so results are reproducible from the same CSVs.

## What's inside

| Page | What it shows |
|------|---------------|
| 🏠 Home | Project pitch, KPI strip, headline result table |
| 📊 Data Exploration | Candlestick + volume + returns distribution per stock |
| 🧪 Feature Engineering | Toggle indicators on the chart + the formula behind each |
| 🤖 Model Comparison | Accuracy, ROC-AUC, ROC curves, confusion matrices, classification reports |
| 🔮 Live Prediction | Pick a date + model → probability of Up + feature snapshot |
| 📓 Code Walkthrough | Notebook cells grouped by pipeline phase (read-only via nbformat) |

## Run locally

```powershell
# from project root (the folder that contains the .ipynb files)
venv\Scripts\Activate.ps1
pip install -r dashboard\requirements.txt

# one-time: build the model artifacts
python dashboard\train_artifacts.py

# launch
streamlit run dashboard\app.py
```

Then open the URL it prints (usually <http://localhost:8501>).

## Deploy to Streamlit Cloud (free)

1. Push the repository to GitHub (make sure `dashboard/` and `Data/` are committed).
2. Go to <https://share.streamlit.io> → **New app**.
3. Repo: pick this repo. Branch: `main`. Main file path: `dashboard/app.py`.
4. Advanced → set the **Python version** to 3.11+.
5. Click **Deploy**. The URL `https://<your-name>-sul-stock-prediction.streamlit.app`
   becomes the link you share on LinkedIn.

If artifact training takes too long on the free tier, commit the generated
`dashboard/artifacts/*.joblib` and `*.json` files to the repo so the cloud
instance skips retraining.

## Constraints honored

- Original `IT_Notebook_*.ipynb` files are **not edited**.
- Feature formulas mirror the notebook cell-by-cell.
- Train/test split, scaler, and target definition follow the notebook exactly:
  chronological 80/20, `StandardScaler` on features only, 1-day or 5-day
  forward direction as the binary target.

## LinkedIn post tips

- Record a 30s screen-capture of the **Live Prediction** page (gauge + feature
  table) — videos crush plain links in feed reach.
- Pin the Streamlit URL as the first comment, not in the post body (LinkedIn
  down-ranks external links in the post itself).
- Tag the libraries used: `#Streamlit #scikitlearn #Plotly #MachineLearning`.
