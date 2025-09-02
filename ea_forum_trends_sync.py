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

# Database connection from environment (supports both local .env and Streamlit Cloud secrets)
DATABASE_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")

if not DATABASE_URL:
    st.error("🔑 Database connection not configured. Please set DATABASE_URL in your environment or Streamlit secrets.")
    st.stop()


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
    """Get database connection with error handling for deployment"""
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.info("💡 Make sure your DATABASE_URL is correctly configured in Streamlit secrets.")
        st.stop()


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


# Custom CSS to reduce header size and margin
st.markdown("""
<style>
    /* Max width only for content areas, not the whole page */
    .main .block-container,
    div[data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    section[data-testid="stMain"] > div > div {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .main > div:first-child > div:first-child h1 {
        font-size: 1.225rem !important; /* 60% smaller than default (~2.5rem -> 1.225rem) */
        margin-top: 0.1rem !important; /* Much smaller top margin */
        margin-bottom: 1rem !important;
    }
    /* Make expander narrower and faster */
    .streamlit-expanderHeader {
        transition: all 0.1s ease !important;
    }
    .streamlit-expanderContent {
        transition: all 0.1s ease !important;
    }
    /* Constrain the features expander width */
    div[data-testid="stExpander"] {
        width: fit-content !important;
        max-width: 650px !important;
    }
    /* Reduce advanced options button padding */
    div[data-testid="stExpander"] > div > div > button {
        padding: 0.25rem 0.5rem !important;
        font-size: 0.9rem !important;
    }
    /* Smaller tip text styling */
    .chart-tip {
        font-size: 0.85em !important;
        padding: 8px 12px !important;
        margin: 8px 0 !important;
    }
    /* Smaller, auto-width dropdown */
    .sort-dropdown select {
        width: 70% !important; /* 30% narrower */
        min-width: auto !important;
        font-size: 0.85rem !important;
        padding: 0.2rem 0.5rem !important;
        cursor: pointer !important;
    }
    /* Vertical alignment for sort controls */
    .sort-controls {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 1rem !important;
        padding: 0.5rem 0 !important;
    }
    .sort-controls > div {
        display: flex !important;
        align-items: center !important;
    }
    .content-label {
        font-size: 1rem !important;
        color: #333 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 2.4rem !important; /* Match dropdown height */
    }
    /* Content card styling */
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
        margin-bottom: 6px;
    }
    .content-meta {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 0px;
    }
    .content-container a { 
        color: inherit; 
        text-decoration: none; 
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 EA Forum Interactive Trends")

# Fetch global stats
global_stats = fetch_ea_global_stats()

if global_stats:
    total_posts = format_content_count(global_stats["total_posts"])
    total_comments = format_content_count(global_stats["total_comments"])
    total_authors = global_stats["total_authors"]

    # Add description with dynamic stats
    st.markdown(
        f"""
        Interactive keyword search and trend analysis for EA Forum data, featuring **{total_posts} posts** and **{total_comments} comments** 
        from **{total_authors} authors**. Search for specific terms, visualize trends over time, and explore related content. Forked from https://www.community-archive.org/.
        """
    )
    
    # Features in collapsible expander (contained width)
    features_container = st.container()
    with features_container:
        with st.expander("ℹ️ Features", expanded=False):
            st.markdown("""
            - 🔍 **Multi-keyword search** with autocomplete suggestions
            - 📊 **Interactive trend visualization** with time-series charts
            - 🎯 **Content discovery** - click and drag on charts to filter results
            - 👥 **Author filtering** for personalized analysis
            - 📈 **Normalized metrics** accounting for forum activity changes
            """)
        
else:
    st.markdown(
        """
        Interactive keyword search and trend analysis for EA Forum data. Search for specific terms, 
        visualize trends over time, and explore related content. Forked from https://www.community-archive.org/.
        """
    )
    
    # Features in collapsible expander (contained width)
    features_container = st.container()
    with features_container:
        with st.expander("ℹ️ Features", expanded=False):
            st.markdown("""
            - 🔍 **Multi-keyword search** with autocomplete suggestions
            - 📊 **Interactive trend visualization** with time-series charts
            - 🎯 **Content discovery** - click and drag on charts to filter results
            - 👥 **Author filtering** for personalized analysis
            - 📈 **Normalized metrics** accounting for forum activity changes
            """)

# Add a divider for visual separation
st.divider()

default_words = ["soil", "alignment", "existential risk", "nematode", "mites"]


def main():
    logging.info("Executing main function")
    
    selection = None

    # Search inputs (single column layout)
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

    # Detect if search words changed and find the most recently added word
    prev_search_words = st.session_state.get("prev_search_words", [])
    recently_added_word = None
    
    if search_words != prev_search_words:
        # Find words that are new (in current but not in previous)
        new_words = [word for word in search_words if word not in prev_search_words]
        if new_words:
            recently_added_word = new_words[-1]  # Get the last added word
    
    # Reorder search_words to put recently added word first, otherwise keep original order (soil first)
    if recently_added_word and recently_added_word in search_words:
        ordered_search_words = [recently_added_word] + [word for word in search_words if word != recently_added_word]
    else:
        ordered_search_words = search_words  # This preserves original order with "soil" first

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

        # Chart section (full width)
        if word_occurrences_dict:
            fig = plot_word_occurrences_ea(
                word_occurrences_dict, monthly_content_counts, normalize
            )
            selection = st.plotly_chart(
                fig,
                use_container_width=True,
                key="word_occurrences",
                selection_mode="box",
                on_select="rerun",
            )
            st.markdown(
                '<div class="chart-tip">💡 <strong>Tip:</strong> Drag horizontally on the graph to filter content below.</div>',
                unsafe_allow_html=True
            )
        else:
            st.write("No data to display. Please enter search words.")

        # Related content section (below chart)
        st.subheader("📚 Related EA Forum Content")
        if search_words:
            # Use recently added word in key to force tab refresh and auto-selection
            tab_key = f"tabs_{recently_added_word or 'default'}_{len(search_words)}"
            tabs = st.tabs(ordered_search_words)
            for word, tab in zip(ordered_search_words, tabs):
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

                    # Add sorting controls with better alignment
                    col_sort1, col_sort2 = st.columns([3, 1])
                    with col_sort1:
                        st.markdown(f'<div class="content-label">Showing content for {word}</div>', unsafe_allow_html=True)
                    with col_sort2:
                        if not word_df.empty:
                            st.markdown('<div class="sort-dropdown">', unsafe_allow_html=True)
                            sort_option = st.selectbox(
                                "",
                                options=["Most Recent", "Most Upvotes", "Most Mentions"],
                                index=2,  # Default to "Most Mentions"
                                key=f"sort_{word}",
                                label_visibility="collapsed"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    if word_df.empty:
                        st.write("No content found")
                    else:
                        # Apply sorting
                        if sort_option == "Most Recent":
                            word_df = word_df.sort_values('created_at', ascending=False)
                        elif sort_option == "Most Upvotes":
                            word_df = word_df.sort_values('score', ascending=False)
                        else:  # Most Mentions
                            # Check if mention_count column exists (for backwards compatibility)
                            if 'mention_count' in word_df.columns:
                                word_df = word_df.sort_values('mention_count', ascending=False)
                            else:
                                # Fallback to score if mention_count not available (cached data)
                                word_df = word_df.sort_values('score', ascending=False)
                                st.warning("⚠️ Cached data doesn't include mention counts. Please refresh to see updated counts.")
                        
                        for _, item in word_df.iterrows():
                                upvotes = item.get('score', 0)
                                mentions = item.get('mention_count', 0)
                                upvote_display = f"{upvotes} upvotes" if upvotes != 1 else "1 upvote"
                                mention_display = f"{mentions} mentions" if mentions != 1 else "1 mention"
                                
                                # Format date to show only the date part
                                date_display = ""
                                if item.get('created_at'):
                                    try:
                                        if isinstance(item['created_at'], str):
                                            date_obj = pd.to_datetime(item['created_at'])
                                        else:
                                            date_obj = item['created_at']
                                        date_display = date_obj.strftime('%Y-%m-%d')
                                    except:
                                        date_display = str(item.get('created_at', ''))[:10]
                                
                                content_type = item.get('content_type', 'post')
                                
                                # Create metadata line with content type first
                                author_name = item.get('author_display_name', 'Unknown')
                                type_display = "Post" if content_type == 'post' else "Comment"
                                metadata_line = f"{type_display} &nbsp;|&nbsp; <b>{author_name}</b> &nbsp;|&nbsp; {date_display} &nbsp;|&nbsp; {upvote_display} &nbsp;|&nbsp; {mention_display}"
                                
                                if content_type == 'comment':
                                    # For comments: show more text (450 chars), smaller font
                                    import re
                                    raw_content = item.get('content', '')
                                    # Strip HTML tags and normalize whitespace
                                    clean_content = re.sub(r'<[^>]+>', '', raw_content)
                                    clean_content = ' '.join(clean_content.split())
                                    content_snippet = clean_content[:450] + "..." if len(clean_content) > 450 else clean_content
                                    post_title = item.get('title', 'Untitled')
                                    metadata_line += f" &nbsp;|&nbsp; Post: {post_title}"
                                    
                                    st.markdown(
                                        f"""
                                        <div class="content-container">
                                            <div class="content-meta" style="font-size: 13px; color: #888; margin-bottom: 8px;">
                                                {metadata_line}
                                            </div>
                                            <div class="content-body" style="font-size: 12pt !important; font-weight: 300 !important; line-height: 1.4 !important;">
                                                <a href="{item.get('url', '#')}" target="_blank" style="color: inherit; text-decoration: none; font-size: 12pt !important; font-weight: 300 !important;">{content_snippet} ↗</a>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    # For posts: larger, bolder text
                                    post_title = item.get('title', 'Untitled')
                                    
                                    st.markdown(
                                        f"""
                                        <div class="content-container">
                                            <div class="content-meta" style="font-size: 13px; color: #888; margin-bottom: 8px;">
                                                {metadata_line}
                                            </div>
                                            <div class="content-header" style="font-size: 16px; font-weight: 600;">
                                                <a href="{item.get('url', '#')}" target="_blank" style="color: inherit; text-decoration: none;">{post_title} ↗</a>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
        else:
            st.write(
                "No search words entered. Please enter words to see related content."
            )


# Run the main function (for Streamlit Cloud deployment)
main()