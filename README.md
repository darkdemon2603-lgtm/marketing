# HANBAO Intelligence Platform

A Streamlit starter for restaurant data upload, customer segmentation, sales/covers forecasting, and natural-language questions.

## Features
- Upload orders/POS, customer profiles, reservations, and product-sales CSVs
- DuckDB in-memory query layer
- Starter customer segments: new/recent, one-time, regular, high-value loyal, at risk, lapsed
- Sales and covers forecasting using weekday and lag features
- Plain-English questions converted into read-only DuckDB SQL using Gemini
- CSV exports for segments and forecasts

## Run
```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Enable AI questions
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key"
```
Or set environment variable `GEMINI_API_KEY`. Create a key at https://aistudio.google.com/apikey. Never commit keys to GitHub.

## Suggested input columns
- Orders/POS: `order_date`, `order_id`, `customer_id`, `revenue`, `covers`
- Customers: `customer_id`, `signup_date`, `email_marketing_consent`, `postcode`
- Reservations: `booking_date`, `customer_id`, `covers`, `status`
- Product sales: `order_date`, `order_id`, `customer_id`, `item`, `quantity`, `unit_price`

Column names are normalized to lowercase snake_case and common alternatives are detected.

## Limitations
- Prototype only, not a full Microsoft Fabric replacement or production CRM.
- Uploaded files live in memory for the current session and are not persisted.
- Unique customers require stable IDs; actual covers should be used when available.
- Forecasts are estimates. More history (ideally 6–24 months) improves seasonal modeling.
- Customer segments use starter heuristics that should be reviewed.
- AI SQL is restricted to read-only query forms, but production deployment still needs authentication, authorization, auditing, data validation, and stronger isolation.
- Production roadmap: persistent warehouse, scheduled ingestion, data quality checks, role-based access, forecast backtesting, reservation/POS reconciliation, campaign attribution.
