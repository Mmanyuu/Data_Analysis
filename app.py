import pandas as pd
import streamlit as st
import altair as alt
import sys
import subprocess
import re
from collections import Counter


@st.cache_data
def load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["published_date"] = (
        pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce")
    return df


st.set_page_config(page_title="SIA Review Pulse", layout="wide")

st.title("SIA Review Pulse")
st.caption("Singapore Airlines review insights from published reviews.")

data_path = "data/singapore_airlines_reviews.csv"
df = load_reviews(data_path)

if df.empty:
    st.warning("No reviews found in the dataset.")
    st.stop()

with st.sidebar:
    st.header("Filters")

    platform_options = sorted(df["published_platform"].dropna().unique())
    selected_platforms = st.multiselect(
        "Platform",
        options=platform_options,
        default=platform_options,
    )

    type_options = sorted(df["type"].dropna().unique())
    selected_types = st.multiselect(
        "Review Type",
        options=type_options,
        default=type_options,
    )

    rating_min, rating_max = int(df["rating"].min()), int(df["rating"].max())
    rating_range = st.slider(
        "Rating Range",
        min_value=rating_min,
        max_value=rating_max,
        value=(rating_min, rating_max),
    )

    date_series = df["published_date"].dropna()
    if date_series.empty:
        st.warning("No valid published dates found in the dataset.")
        st.stop()
    date_min = date_series.min().date()
    date_max = date_series.max().date()
    default_end_date = date_max
    default_start_date = max(
        date_min,
        (pd.Timestamp(date_max) - pd.DateOffset(months=12)).date(),
    )
    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=date_min,
        max_value=date_max,
    )
    end_date = st.date_input(
        "End Date",
        value=default_end_date,
        min_value=date_min,
        max_value=date_max,
    )


filtered = df.copy()
if selected_platforms:
    filtered = filtered[filtered["published_platform"].isin(selected_platforms)]
if selected_types:
    filtered = filtered[filtered["type"].isin(selected_types)]
filtered = filtered[filtered["rating"].between(rating_range[0], rating_range[1])]

if isinstance(start_date, pd.Timestamp):
    start_date = start_date.date()
if isinstance(end_date, pd.Timestamp):
    end_date = end_date.date()
if start_date > end_date:
    start_date, end_date = end_date, start_date
filtered = filtered[
    (filtered["published_date"].dt.date >= start_date)
    & (filtered["published_date"].dt.date <= end_date)
]

total_reviews = len(filtered)
avg_rating = filtered["rating"].mean()
positive_share = (
    (filtered["rating"].between(4, 5)).mean() * 100 if total_reviews else 0
)
negative_share = (
    (filtered["rating"].between(1, 2)).mean() * 100 if total_reviews else 0
)

if total_reviews:
    avg_text = f"{avg_rating:.2f}"
    positive_text = f"{positive_share:.1f}%"
    negative_text = f"{negative_share:.1f}%"
else:
    avg_text = "N/A"
    positive_text = "0.0%"
    negative_text = "0.0%"

carousel_html = f"""
<style>
.summary-carousel {{
  background-color: #fff3cd;
  color: #000;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  margin-bottom: 0.75rem;
  overflow: hidden;
}}
.summary-track {{
  display: flex;
  width: 400%;
  animation: summary-slide 12s infinite;
}}
.summary-card {{
  flex: 0 0 25%;
  text-align: center;
  font-weight: 600;
}}
.summary-card strong {{
  font-weight: 800;
}}
@keyframes summary-slide {{
  0% {{ transform: translateX(0%); }}
  20% {{ transform: translateX(0%); }}
  25% {{ transform: translateX(-25%); }}
  45% {{ transform: translateX(-25%); }}
  50% {{ transform: translateX(-50%); }}
  70% {{ transform: translateX(-50%); }}
  75% {{ transform: translateX(-75%); }}
  95% {{ transform: translateX(-75%); }}
  100% {{ transform: translateX(0%); }}
}}
</style>
<div class="summary-carousel">
  <div class="summary-track">
    <div class="summary-card">Total reviews <strong>{total_reviews:,}</strong></div>
    <div class="summary-card">Average rating <strong>{avg_text}</strong></div>
    <div class="summary-card">Positive <strong>{positive_text}</strong></div>
    <div class="summary-card">Negative <strong>{negative_text}</strong></div>
  </div>
</div>
"""

st.markdown(carousel_html, unsafe_allow_html=True)

st.subheader("Review Snapshot")

helpful_share = (
    (filtered["helpful_votes"] > 0).mean() * 100 if total_reviews else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Average Rating", f"{avg_rating:.2f}" if total_reviews else "N/A")
col3.metric("Helpful Review Share", f"{helpful_share:.1f}%")

st.subheader("Ratings Distribution")
rating_counts = (
    filtered.dropna(subset=["rating"])
    .groupby("rating")
    .size()
    .reset_index(name="count")
)

ratings_chart = (
    alt.Chart(rating_counts)
    .mark_bar(color="#1f77b4")
    .encode(
        x=alt.X("rating:O", title="Rating"),
        y=alt.Y("count:Q", title="Reviews"),
        tooltip=["rating", "count"],
    )
    .properties(height=260)
)
st.altair_chart(ratings_chart, use_container_width=True)

st.subheader("Reviews Over Time")
time_series = (
    filtered.dropna(subset=["published_date"])
    .assign(month=lambda x: x["published_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")
    .size()
    .reset_index(name="count")
)

time_chart = (
    alt.Chart(time_series)
    .mark_line(point=True, color="#d95f02")
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("count:Q", title="Reviews"),
        tooltip=["month:T", "count"],
    )
    .properties(height=260)
)
st.altair_chart(time_chart, use_container_width=True)

st.subheader("Rating Trends Over Time")
trend_granularity = st.selectbox(
    "Trend Granularity",
    options=["Monthly", "Quarterly"],
    index=0,
)

trend_df = filtered.dropna(subset=["published_date", "rating"]).copy()
if trend_granularity == "Quarterly":
    trend_df = trend_df.assign(
        period=lambda x: x["published_date"].dt.to_period("Q").dt.to_timestamp()
    )
else:
    trend_df = trend_df.assign(
        period=lambda x: x["published_date"].dt.to_period("M").dt.to_timestamp()
    )

rating_trend = (
    trend_df.groupby("period")["rating"]
    .mean()
    .reset_index(name="avg_rating")
)

trend_chart = (
    alt.Chart(rating_trend)
    .mark_line(point=True, color="#2ca02c")
    .encode(
        x=alt.X("period:T", title="Period"),
        y=alt.Y("avg_rating:Q", title="Average Rating"),
        tooltip=["period:T", "avg_rating:Q"],
    )
    .properties(height=260)
)
st.altair_chart(trend_chart, use_container_width=True)

st.subheader("Latest Reviews")
display_cols = [
    "published_date",
    "published_platform",
    "rating",
    "type",
    "title",
    "text",
    "helpful_votes",
]
st.dataframe(
    filtered.sort_values("published_date", ascending=False)[display_cols].head(10),
    use_container_width=True,
)

st.subheader("Positive Review Keywords (Ratings 4-5)")
st.caption("Highlights common themes from higher-rated reviews.")

try:
    from wordcloud import WordCloud, STOPWORDS
except ImportError:
    with st.status("Installing wordcloud...", expanded=False):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
    from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.update(
    {
        "flight",
        "airline",
        "singapore",
        "airlines",
        "sia",
        "air",
        "plane",
        "crew",
        "seat",
        "seats",
        "service",
        "staff",
    }
)

def extract_top_keywords(texts, stopwords_set, top_n=10):
    tokens = []
    for text in texts:
        words = re.findall(r"[A-Za-z]{3,}", text.lower())
        tokens.extend([w for w in words if w not in stopwords_set])
    return Counter(tokens).most_common(top_n)

def keyword_context_snippets(texts, keyword, max_snippets=3):
    snippets = []
    pattern = re.compile(rf"(.{{0,60}}\\b{re.escape(keyword)}\\b.{{0,60}})", re.IGNORECASE)
    for text in texts:
        match = pattern.search(text)
        if match:
            snippets.append(match.group(1).strip())
        if len(snippets) >= max_snippets:
            break
    return snippets

positive_text = " ".join(
    filtered.loc[filtered["rating"].between(4, 5), ["title", "text"]]
    .fillna("")
    .agg(" ".join, axis=1)
    .tolist()
)
positive_texts = filtered.loc[filtered["rating"].between(4, 5), ["title", "text"]].fillna("").agg(" ".join, axis=1).tolist()
if positive_text.strip():
    positive_wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        stopwords=stopwords,
    ).generate(positive_text)
    st.image(positive_wc.to_array(), use_container_width=True)
else:
    st.info("No positive reviews available for the current filters.")

st.subheader("Negative Review Keywords (Ratings 1-2)")
st.caption("Highlights common themes from lower-rated reviews.")

negative_text = " ".join(
    filtered.loc[filtered["rating"].between(1, 2), ["title", "text"]]
    .fillna("")
    .agg(" ".join, axis=1)
    .tolist()
)
negative_texts = filtered.loc[filtered["rating"].between(1, 2), ["title", "text"]].fillna("").agg(" ".join, axis=1).tolist()
if negative_text.strip():
    negative_wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        stopwords=stopwords,
    ).generate(negative_text)
    st.image(negative_wc.to_array(), use_container_width=True)
else:
    st.info("No negative reviews available for the current filters.")

st.subheader("Keyword Context (Top 10)")
st.caption("Toggle between positive and negative keywords to see sample context.")

tab_positive, tab_negative = st.tabs(["Positive (4-5)", "Negative (1-2)"])

with tab_positive:
    top_positive = extract_top_keywords(positive_texts, stopwords, top_n=10)
    if top_positive:
        for keyword, count in top_positive:
            with st.expander(f"{keyword} ({count})"):
                snippets = keyword_context_snippets(positive_texts, keyword)
                if snippets:
                    for snippet in snippets:
                        st.write(f"- {snippet}")
                else:
                    st.write("No matching snippets found.")
    else:
        st.info("No positive keywords available for the current filters.")

with tab_negative:
    top_negative = extract_top_keywords(negative_texts, stopwords, top_n=10)
    if top_negative:
        for keyword, count in top_negative:
            with st.expander(f"{keyword} ({count})"):
                snippets = keyword_context_snippets(negative_texts, keyword)
                if snippets:
                    for snippet in snippets:
                        st.write(f"- {snippet}")
                else:
                    st.write("No matching snippets found.")
    else:
        st.info("No negative keywords available for the current filters.")
