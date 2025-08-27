import streamlit as st
from streamlit_tags import st_tags

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, date
import plotly.express as px
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression

load_dotenv(".env")
st.set_page_config(layout="wide")

# Add logging
logging.basicConfig(level=logging.INFO)

# Database connection from environment
DATABASE_URL = os.getenv("DATABASE_URL")


def format_content_count(count):
    """Format content count for display"""
    digits = len(str(int(count)))
    if digits > 9:
        return f"{count/10**9:.1f}B"
    elif digits > 6:
        return f"{count/10**6:.1f}M"
    elif digits > 3:
        return f"{count/10**3:.1f}K"
    return str(count)


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)


@st.cache_data(ttl=3600)
def fetch_ea_content_cached(search_query, start_date, end_date, limit=100):
    """Cache wrapper for content search"""
    logging.info(f"Executing fetch_ea_content_cached for query: {search_query}")
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM search_ea_content(%s, %s, %s, %s)",
                (search_query, start_date, end_date, limit)
            )
            results = cur.fetchall()
            return [dict(record) for record in results]


def fetch_ea_content(search_words, start_date, end_date, limit=100):
    """Fetch EA Forum content for multiple search words"""
    logging.info(f"Executing fetch_ea_content for words: {search_words}")
    results = []
    for word in search_words:
        result = fetch_ea_content_cached(word, start_date, end_date, limit)
        # Add search word to each result
        for item in result:
            item["search_word"] = word
        results.extend(result)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


@st.cache_data(ttl=3600)
def fetch_word_occurrences_ea_cached(word, start_date, end_date, author_ids):
    """Cache wrapper for word occurrences"""
    logging.info(f"Executing fetch_word_occurrences_ea_cached for word: {word}")
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM word_occurrences_ea(%s, %s)",
                (word, author_ids if author_ids else None)
            )
            results = cur.fetchall()
            # Filter by date range
            filtered = [
                dict(record) for record in results
                if start_date <= datetime.strptime(record["month"], "%Y-%m").date() <= end_date
            ]
            return {word: filtered}


def fetch_word_occurrences_ea(search_words, start_date, end_date, author_ids):
    """Fetch word occurrences for multiple words"""
    logging.info(f"Executing fetch_word_occurrences_ea for words: {search_words}")
    results = {}
    for word in search_words:
        word_result = fetch_word_occurrences_ea_cached(word, start_date, end_date, author_ids)
        results.update(word_result)
    return results


@st.cache_data(ttl=3600)
def fetch_monthly_ea_content_counts():
    """Get monthly content counts for normalization"""
    logging.info("Executing fetch_monthly_ea_content_counts")
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM get_monthly_ea_content_counts()")
            results = cur.fetchall()
            
    data = [dict(record) for record in results]
    df = pd.DataFrame(data)
    if not df.empty:
        df["month"] = pd.to_datetime(df["month"], utc=True)
    return df


def plot_word_occurrences_ea(word_occurrences_dict, monthly_content_counts, normalize):
    """Plot word occurrences over time for EA Forum data"""
    logging.info("Executing plot_word_occurrences_ea")
    df_list = []
    for word, result in word_occurrences_dict.items():
        if result:  # Check if result not empty
            df = pd.DataFrame(result)
            df["month"] = pd.to_datetime(df["month"], utc=True)
            df["word"] = word
            df_list.append(df)

    if not df_list:  # If no data, return empty figure
        return go.Figure()

    df = pd.concat(df_list)
    df = df.merge(monthly_content_counts, on="month", how="left")

    if normalize:
        df["normalized_count"] = df["word_count"] / df["content_count"] * 1000
        y_col, y_title = "normalized_count", "Occurrences per 1000 posts/comments"
    else:
        y_col, y_title = "word_count", "Word Count"

    fig = px.line(
        df,
        x="month",
        y=y_col,
        color="word",
        title=f'Word Occurrences Over Time {"(normalized)" if normalize else ""}',
    )
    fig.update_layout(xaxis_title="Month", yaxis_title=y_title)
    fig.update_traces(mode="lines+markers")  # Add markers for selection
    return fig


@st.cache_data(ttl=3600)
def fetch_ea_authors():
    """Get list of EA Forum authors for filtering"""
    logging.info("Executing fetch_ea_authors")
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM get_ea_authors()")
            results = cur.fetchall()
            return [dict(record) for record in results]


@st.cache_data(ttl=3600)
def fetch_ea_global_stats():
    """Get global EA Forum statistics"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM get_ea_global_stats()")
            result = cur.fetchone()
            return dict(result) if result else None


st.title("📈 EA Forum Interactive Trends")

# Fetch global stats
global_stats = fetch_ea_global_stats()

if global_stats:
    total_posts = format_content_count(global_stats["total_posts"])
    total_comments = format_content_count(global_stats["total_comments"])
    total_authors = global_stats["total_authors"]

    # Add explanation with dynamic stats
    st.markdown(
        f"""
        Interactive keyword search and trend analysis for EA Forum data, featuring **{total_posts} posts** and **{total_comments} comments** 
        from **{total_authors} authors**. Search for specific terms, visualize trends over time, and explore related content.
        
        **Features:**
        - 🔍 **Multi-keyword search** with autocomplete suggestions
        - 📊 **Interactive trend visualization** with time-series charts
        - 🎯 **Content discovery** - click and drag on charts to filter results
        - 👥 **Author filtering** for personalized analysis
        - 📈 **Normalized metrics** accounting for forum activity changes
        """
    )
else:
    st.markdown(
        """
        Interactive keyword search and trend analysis for EA Forum data. Search for specific terms, 
        visualize trends over time, and explore related content.
        """
    )

# Add a divider for visual separation
st.divider()

default_words = ["alignment", "safety", "existential risk"]


def main():
    logging.info("Executing main function")
    
    selection = None
    col1, col2 = st.columns(2)

    with col1:
        search_words = st_tags(
            label="Enter search words",
            text="Press enter after each word",
            value=default_words,
            suggestions=[
                "longtermism", "x-risk", "AI safety", "effective altruism", 
                "global health", "animal welfare", "biosecurity", "alignment",
                "superintelligence", "AGI", "pandemic", "climate change",
                "malaria", "deworming", "factory farming", "cultured meat"
            ],
            maxtags=10,
            key="search_words",
        )

        # Move advanced options to an expander
        with st.expander("🔧 Advanced Options"):
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", value=date(2020, 1, 1))
            with date_col2:
                end_date = st.date_input("End Date", value=date.today())

            authors = fetch_ea_authors()
            author_options = {
                author["author_display_name"]: author["author_id"] 
                for author in authors[:100]  # Limit to first 100 for performance
            }
            selected_authors = st.multiselect(
                "Filter by Authors", options=list(author_options.keys())
            )
            author_ids = [author_options[author] for author in selected_authors]
            normalize = st.checkbox("Normalize by monthly content count", value=True)

    # Check if query parameters have changed
    query_changed = (
        "prev_search_words" not in st.session_state
        or "prev_start_date" not in st.session_state
        or "prev_end_date" not in st.session_state
        or "prev_author_ids" not in st.session_state
        or search_words != st.session_state.get("prev_search_words")
        or start_date != st.session_state.get("prev_start_date")
        or end_date != st.session_state.get("prev_end_date")
        or author_ids != st.session_state.get("prev_author_ids")
    )

    if query_changed or "content_df" not in st.session_state:
        if search_words:
            with st.spinner("Fetching EA Forum data..."):
                st.session_state.content_df = fetch_ea_content(search_words, start_date, end_date)
                st.session_state.word_occurrences_dict = fetch_word_occurrences_ea(search_words, start_date, end_date, author_ids)
                st.session_state.monthly_content_counts = fetch_monthly_ea_content_counts()

                # Update previous query parameters
                st.session_state.update(
                    {
                        "prev_search_words": search_words,
                        "prev_start_date": start_date,
                        "prev_end_date": end_date,
                        "prev_author_ids": author_ids,
                    }
                )
        else:
            st.session_state.content_df = pd.DataFrame()
            st.session_state.word_occurrences_dict = {}
            st.session_state.monthly_content_counts = fetch_monthly_ea_content_counts()

    if "content_df" in st.session_state:
        content_df = st.session_state.content_df
        word_occurrences_dict = st.session_state.word_occurrences_dict
        monthly_content_counts = st.session_state.monthly_content_counts

        with col1:
            st.subheader("📊 Keyword Trends")
            if word_occurrences_dict:
                fig = plot_word_occurrences_ea(
                    word_occurrences_dict, monthly_content_counts, normalize
                )
                st.info(
                    "💡 **Tip:** Drag horizontally on the graph to filter content in the right column."
                )
                selection = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="word_occurrences",
                    selection_mode="box",
                    on_select="rerun",
                )
            else:
                st.write("No data to display. Please enter search words.")

        with col2:
            st.subheader("📚 Related EA Forum Content")
            content_container = st.container()
            content_container.markdown(
                """
                <style>
                [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                    height: 80vh;
                    overflow-y: auto;
                }
                .content-container {
                    display: flex;
                    flex-direction: column;
                    margin-bottom: 20px;
                    padding: 15px;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                }
                .content-header {
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .content-meta {
                    font-size: 0.9em;
                    color: #666;
                    margin-bottom: 10px;
                }
                .content-body {
                    line-height: 1.5;
                }
                .content-container a { color: inherit; text-decoration: none; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            with content_container:
                if search_words:
                    tabs = st.tabs(search_words)
                    for word, tab in zip(search_words, tabs):
                        with tab:
                            if selection and selection["selection"]["points"]:
                                selected_dates = [
                                    pd.to_datetime(point["x"])
                                    for point in selection["selection"]["points"]
                                ]
                                start_date_filter = min(selected_dates).date()
                                end_date_filter = max(selected_dates).date()
                                word_content = fetch_ea_content_cached(
                                    word, start_date_filter, end_date_filter
                                )
                                word_df = pd.DataFrame(word_content)
                                if not word_df.empty:
                                    word_df["created_at"] = pd.to_datetime(word_df["created_at"], utc=True)
                            else:
                                if "search_word" in content_df.columns:
                                    word_df = content_df[
                                        content_df["search_word"] == word
                                    ]
                                else:
                                    st.error(
                                        "'search_word' column not found in content DataFrame."
                                    )
                                    word_df = pd.DataFrame()

                            st.write(f"Showing content for **'{word}'**")
                            if word_df.empty:
                                st.write("No content found")
                            else:
                                for _, item in word_df.iterrows():
                                    # Highlight the search word in content
                                    content_text = item.get("content", "")[:500] + "..." if len(item.get("content", "")) > 500 else item.get("content", "")
                                    highlighted_text = content_text.replace(
                                        word, f"<b>{word}</b>"
                                    ) if content_text else ""
                                    
                                    content_type_badge = "📝 Post" if item["content_type"] == "post" else "💬 Comment"
                                    score_display = f"⭐ {item.get('score', 0)}" if item.get('score') else ""
                                    
                                    st.markdown(
                                        f"""
                                        <div class="content-container">
                                            <div class="content-header">
                                                {content_type_badge} <a href="{item.get('url', '#')}" target="_blank" style="color: inherit; text-decoration: none;">{item.get('title', 'Untitled')}</a>
                                            </div>
                                            <div class="content-meta">
                                                <b>{item.get('author_display_name', 'Unknown')}</b> - {item.get('created_at', '')} {score_display}
                                            </div>
                                            <div class="content-body">
                                                {highlighted_text}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                else:
                    st.write(
                        "No search words entered. Please enter words to see related content."
                    )


if __name__ == "__main__":
    main()