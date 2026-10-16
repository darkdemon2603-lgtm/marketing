import os, re
import numpy as np
import pandas as pd
import streamlit as st
import duckdb

st.set_page_config(page_title="HANBAO Intelligence", page_icon="🍔", layout="wide")
st.title("HANBAO Intelligence")
st.caption("Customer intelligence · Sales and covers forecasting · Ask your data")

def clean(df):
    df = df.copy()
    df.columns = [re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df

def col(df, names):
    return next((n for n in names if n in df.columns), None)

def find_cols(df):
    return {
        "date": col(df, ["order_date","date","created_at","transaction_date","visit_date","booking_date"]),
        "sales": col(df, ["revenue","sales","net_sales","order_total","total","amount","total_spend","spend"]),
        "customer": col(df, ["customer_id","guest_id","profile_id","email","customer","guest"]),
        "order": col(df, ["order_id","transaction_id","check_id","ticket_id","booking_id"]),
        "covers": col(df, ["covers","cover_count","guests","guest_count","party_size","number_of_guests"]),
    }

def aggregate_daily(df, c):
    if not c["date"]:
        raise ValueError("Date column not found. Try order_date, date, visit_date, or booking_date.")
    x = df.copy()
    x[c["date"]] = pd.to_datetime(x[c["date"]], errors="coerce").dt.floor("D")
    x = x.dropna(subset=[c["date"]])
    if x.empty:
        raise ValueError("No valid dates found.")
    x["_sales"] = pd.to_numeric(x[c["sales"]], errors="coerce").fillna(0) if c["sales"] else 0.0
    if c["covers"]:
        x["_covers"] = pd.to_numeric(x[c["covers"]], errors="coerce").fillna(0)
    elif c["order"]:
        x["_covers"] = x[c["order"]]
    else:
        x["_covers"] = 1
    daily = x.groupby(c["date"]).agg(sales=("_sales","sum"), covers=("_covers","sum")).sort_index()
    if c["covers"] is None and c["order"]:
        daily["covers"] = x.groupby(c["date"])[c["order"]].nunique()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(idx, fill_value=0).rename_axis("date")

def segment_customers(df):
    c = find_cols(df)
    if not c["customer"] or not c["date"]:
        return None, "Customer segmentation requires customer_id (or equivalent) and a date column."
    x = df.copy()
    x[c["date"]] = pd.to_datetime(x[c["date"]], errors="coerce")
    x = x.dropna(subset=[c["date"], c["customer"]])
    if x.empty: return None, "No valid customer IDs and dates found."
    x["_spend"] = pd.to_numeric(x[c["sales"]], errors="coerce").fillna(0) if c["sales"] else 0
    ref = x[c["date"]].max().normalize() + pd.Timedelta(days=1)
    agg = x.groupby(c["customer"]).agg(last_visit=(c["date"],"max"), first_visit=(c["date"],"min"), total_spend=("_spend","sum"))
    if c["order"]:
        visits = x.groupby(c["customer"])[c["order"]].nunique()
    else:
        visits = x.groupby(c["customer"]).size()
    agg["visits"] = visits
    agg["recency_days"] = (ref - agg["last_visit"].dt.normalize()).dt.days
    agg["avg_spend_per_visit"] = agg["total_spend"] / agg["visits"].clip(lower=1)
    agg["segment"] = "Regular"
    agg.loc[agg["visits"] == 1, "segment"] = "One-time"
    agg.loc[(agg["visits"] == 1) & (agg["recency_days"] <= 30), "segment"] = "New / recent"
    agg.loc[(agg["visits"] >= 3) & (agg["recency_days"] <= 45) & (agg["total_spend"] >= agg["total_spend"].quantile(.75)), "segment"] = "High-value loyal"
    agg.loc[(agg["visits"] >= 2) & (agg["recency_days"] > 60) & (agg["recency_days"] <= 120), "segment"] = "At risk"
    agg.loc[agg["recency_days"] > 120, "segment"] = "Lapsed"
    return agg.reset_index().rename(columns={c["customer"]:"customer_id"}), None

def forecast(daily, horizon):
    from sklearn.ensemble import RandomForestRegressor
    future_dates = pd.date_range(daily.index.max()+pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.DataFrame(index=future_dates)
    for metric in ["sales","covers"]:
        s = daily[metric].astype(float)
        if len(s) < 35:
            means = s.groupby(s.index.dayofweek).mean()
            out[metric] = [max(0, float(means.get(d.dayofweek, s.mean()))) for d in future_dates]
            continue
        X, y = [], []
        def feat(ds, hist):
            f = {"dow":ds.dayofweek, "day":ds.day, "month":ds.month, "week":int(ds.isocalendar().week),
                 "weekend":int(ds.dayofweek >= 4), "trend":(ds-hist.index.min()).days}
            for lag in [1,7,14,21,28]:
                f[f"lag{lag}"] = float(hist.iloc[-lag]) if len(hist)>=lag else float(hist.mean())
            for w in [7,14,28]:
                f[f"avg{w}"] = float(hist.tail(min(w,len(hist))).mean())
            return f
        for i in range(28,len(s)):
            X.append(feat(s.index[i],s.iloc[:i])); y.append(float(s.iloc[i]))
        X = pd.DataFrame(X).fillna(0)
        model = RandomForestRegressor(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(X,y)
        hist = s.copy(); preds=[]
        for d in future_dates:
            row = pd.DataFrame([feat(d,hist)]).reindex(columns=X.columns,fill_value=0)
            p=max(0,float(model.predict(row)[0])); preds.append(p); hist.loc[d]=p
        out[metric]=preds
    out["covers"] = out["covers"].round().astype(int)
    out["sales"] = out["sales"].round(2)
    out.index.name="date"
    return out

def safe_sql(sql):
    sql = (sql or "").strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.I).strip()
    if not re.match(r"^(select|with)\b", sql, re.I): raise ValueError("Only SELECT queries are allowed.")
    if ";" in sql.rstrip(";"): raise ValueError("Only one SQL statement is allowed.")
    if re.search(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|call|execute|vacuum|set|reset)\b", sql, re.I):
        raise ValueError("Restricted SQL operation detected.")
    return sql.rstrip(";")

st.sidebar.header("Upload your data")
uploaders = {
    "orders": st.sidebar.file_uploader("Orders / POS CSV", type="csv"),
    "customers": st.sidebar.file_uploader("Customer profiles CSV (optional)", type="csv"),
    "reservations": st.sidebar.file_uploader("Bookings / reservations CSV (optional)", type="csv"),
    "products": st.sidebar.file_uploader("Product sales CSV (optional)", type="csv"),
}
tables={}
for name,file in uploaders.items():
    if file:
        try: tables[name]=clean(pd.read_csv(file))
        except Exception as e: st.sidebar.error(f"{name}: {e}")

if not tables:
    st.markdown("""
    ### Upload your data to get started
    Start with an orders/POS CSV. Optional uploads can include customer profiles, reservations and product sales.

    **Recommended columns**
    - Orders: `order_date`, `order_id`, `customer_id`, `revenue`, `covers`
    - Customers: `customer_id`, `signup_date`, `email_marketing_consent`, `postcode`
    - Reservations: `booking_date`, `customer_id`, `covers`, `status`
    - Product sales: `order_date`, `order_id`, `customer_id`, `item`, `quantity`, `unit_price`

    Stable customer IDs are required for reliable unique-customer segmentation. Actual covers are preferable to using order count as a proxy.
    """)
    st.stop()

con=duckdb.connect(":memory:")
for name,df in tables.items(): con.register(name,df)
orders=tables.get("orders")
daily=None
if orders is not None:
    st.sidebar.success("Loaded: "+", ".join(f"{k} ({len(v):,} rows)" for k,v in tables.items()))
    st.header("1. Performance overview")
    try:
        daily=aggregate_daily(orders,find_cols(orders))
        a,b,c,d=st.columns(4)
        a.metric("Historical sales",f"£{daily.sales.sum():,.2f}")
        b.metric("Covers / visit proxy",f"{daily.covers.sum():,.0f}")
        c.metric("Days of history",f"{len(daily):,}")
        d.metric("Average daily sales",f"£{daily.sales.mean():,.2f}")
        st.line_chart(daily[["sales"]])
    except Exception as e: st.warning(f"Daily metrics unavailable: {e}")
else:
    st.info("Upload an orders/POS CSV to enable performance and forecasting.")

st.header("2. Customer segmentation")
if orders is not None:
    seg,err=segment_customers(orders)
    if err: st.warning(err)
    else:
        a,b,c=st.columns(3)
        a.metric("Identified customers",f"{seg.customer_id.nunique():,}")
        b.metric("High-value loyal",f"{(seg.segment=='High-value loyal').sum():,}")
        c.metric("At risk / lapsed",f"{seg.segment.isin(['At risk','Lapsed']).sum():,}")
        summary=seg.groupby("segment",as_index=False).agg(customers=("customer_id","nunique"),revenue=("total_spend","sum"),avg_visits=("visits","mean"),avg_recency_days=("recency_days","mean"))
        x,y=st.columns([1,2])
        with x: st.bar_chart(summary.set_index("segment")["customers"])
        with y: st.dataframe(summary,use_container_width=True,hide_index=True)
        choices=st.multiselect("Filter customer segments",sorted(seg.segment.unique()),default=sorted(seg.segment.unique()))
        filtered=seg[seg.segment.isin(choices)]
        st.dataframe(filtered,use_container_width=True,hide_index=True)
        st.download_button("Download customer segments CSV",filtered.to_csv(index=False).encode(), "hanbao_customer_segments.csv","text/csv")
        st.caption("Starter rule-based segments; validate thresholds against your actual business definitions.")

st.header("3. Predict sales and covers per day")
if daily is not None and len(daily)>=7:
    horizon=st.selectbox("Forecast horizon", [7,14,28,56], index=1, format_func=lambda n:f"{n} days")
    if st.button("Generate sales & covers forecast",type="primary"):
        try:
            with st.spinner("Training forecasting models..."):
                fc=forecast(daily,int(horizon))
            left,right=st.columns(2)
            with left:
                st.write("**Predicted sales**")
                st.line_chart(fc[["sales"]])
                st.metric("Forecast sales total",f"£{fc.sales.sum():,.2f}")
            with right:
                st.write("**Predicted covers / visits**")
                st.line_chart(fc[["covers"]])
                st.metric("Forecast covers total",f"{fc.covers.sum():,.0f}")
            st.dataframe(fc.reset_index(),use_container_width=True,hide_index=True)
            st.download_button("Download forecast CSV",fc.reset_index().to_csv(index=False).encode(),"hanbao_sales_covers_forecast.csv","text/csv")
        except Exception as e: st.error(f"Forecast failed: {type(e).__name__}: {e}")
else:
    st.info("Upload at least 7 days of dated sales data. Six to 24 months is preferable for seasonality and more reliable forecasts.")

st.header("4. Ask HANBAO Intelligence")
question=st.text_input("Ask a question about your data",placeholder="Which customer segment has the highest spend? What are expected Friday sales?")
if st.button("Ask your data"):
    if not question.strip(): st.warning("Enter a question first.")
    else:
        try:
            key = st.secrets.get("GEMINI_API_KEY","").strip()
        except Exception: key = ""
        key = key or os.getenv("GEMINI_API_KEY","").strip()
        if not key:
            st.warning("Configure GEMINI_API_KEY in Streamlit Secrets or your environment to enable natural-language questions.")
            st.code('GEMINI_API_KEY = "your-gemini-api-key"',language="toml")
        else:
            try:
                from google import genai
                client=genai.Client(api_key=key)
                schema="\n".join(f"Table {n}: {', '.join(map(str,df.columns))}" for n,df in tables.items())
                prompt=f"""Convert the user's question into one read-only DuckDB SQL query.
Available tables/columns:
{schema}
Rules: return SQL only; use only listed tables/columns; one SELECT or WITH...SELECT; no modifications.
If data is insufficient, return SELECT 'Insufficient data: explain which fields are missing' AS answer.
Question: {question}"""
                with st.spinner("Thinking..."):
                    response=client.interactions.create(model="gemini-3.8-flash", input=prompt)
                sql=safe_sql(response.output_text)
                st.write("**SQL used**"); st.code(sql,language="sql")
                result=con.execute(sql).df()
                st.write("**Answer**"); st.dataframe(result,use_container_width=True,hide_index=True)
            except Exception as e: st.error(f"Question failed: {type(e).__name__}: {e}")

st.divider()
st.caption("Prototype: uploads are held in memory for the current session. This is not yet a persistent warehouse or production CRM. Forecasts are estimates and should be backtested before operational use.")
