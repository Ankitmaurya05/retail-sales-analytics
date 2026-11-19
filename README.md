# Retail Sales Analytics & Demand Forecasting

An end-to-end portfolio project built with Python, Pandas, Plotly, Streamlit, and Scikit-learn.

## Features
- Interactive KPI dashboard for revenue, orders, profit, and units sold
- Date, category, and region filters
- Product/category performance analysis
- Monthly sales trends and regional comparisons
- 30-day demand forecast using a Random Forest regression model with lag and calendar features
- Upload your own retail CSV and download forecast results
- Synthetic demo data generator (no private/customer data)

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/
2. Connect GitHub and select this repository.
3. Choose branch `main` and main file `app.py`.
4. Deploy.

## CSV format
Required columns: `date`, `product`, `category`, `region`, `units_sold`, `unit_price`, `unit_cost`.
The app also accepts `revenue` instead of `unit_price` when revenue is already calculated per row.

## Model notes
The app aggregates daily sales and trains a Random Forest model using lagged sales and calendar features. The demo data is synthetic and results are illustrative, not business forecasts. For a real business dataset, validate with a chronological holdout before operational use.

## Resume description
**Retail Sales Analytics & Demand Forecasting | Python, Pandas, Scikit-learn, Plotly, Streamlit**
- Built an interactive dashboard to explore retail revenue, profit, product performance, and regional sales.
- Developed a machine-learning forecasting workflow using lag features and calendar variables.
- Added CSV upload, data validation, interactive filters, and downloadable forecast output.
- Deployed the application with Streamlit Community Cloud.

## License
MIT
