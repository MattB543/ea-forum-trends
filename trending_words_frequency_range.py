import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, date
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

load_dotenv(".env")
st.set_page_config(layout="wide")

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)

@st.cache_data(ttl=3600)
def fetch_monthly_ea_content_counts():
    """Get monthly content counts for normalization"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM get_monthly_ea_content_counts()")
            results = cur.fetchall()
            
    data = [dict(record) for record in results]
    df = pd.DataFrame(data)
    if not df.empty:
        df["month"] = pd.to_datetime(df["month"], utc=True)
    return df

@st.cache_data(ttl=3600)
def fetch_word_occurrences_ea_cached(word, start_date, end_date, author_ids):
    """Cache wrapper for word occurrences"""
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

def calculate_trend_metrics(word_data, monthly_counts):
    """Calculate comprehensive trend metrics for a word's time series data"""
    if not word_data:
        return None
    
    # Create DataFrame from word occurrence data
    df = pd.DataFrame(word_data)
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
    df = df.sort_values('month')
    
    # Merge with monthly totals for normalization
    monthly_df = monthly_counts.copy()
    monthly_df['month'] = monthly_df['month'].dt.to_period('M').dt.to_timestamp()
    df['month'] = df['month'].dt.to_period('M').dt.to_timestamp()
    
    df = df.merge(monthly_df, on='month', how='left')
    df['normalized_count'] = df['word_count'] / df['content_count'] * 1000
    
    if len(df) < 3:  # Need at least 3 points for meaningful trend analysis
        return None
    
    # Prepare data for regression
    df['month_numeric'] = (df['month'] - df['month'].min()).dt.days
    X = df['month_numeric'].values.reshape(-1, 1)
    y_norm = df['normalized_count'].values
    
    # Linear regression for normalized counts  
    reg_norm = LinearRegression().fit(X, y_norm)
    slope_norm = reg_norm.coef_[0]
    r2_norm = reg_norm.score(X, y_norm)
    
    # Statistical significance test for normalized trend
    y_pred = reg_norm.predict(X)
    residuals = y_norm - y_pred
    n = len(y_norm)
    if n > 2:
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / np.sum((X.flatten() - X.mean())**2))
        t_stat = slope_norm / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    else:
        t_stat = 0
        p_value = 1.0
    
    # Growth rate calculations
    recent_period = df.tail(3)['normalized_count'].mean()
    early_period = df.head(3)['normalized_count'].mean()
    
    if early_period > 0:
        percentage_growth = ((recent_period - early_period) / early_period) * 100
        # Compound monthly growth rate
        n_months = len(df) - 1
        if n_months > 0 and recent_period > 0 and early_period > 0:
            cagr = (pow(recent_period / early_period, 1/n_months) - 1) * 100
        else:
            cagr = 0
    else:
        percentage_growth = float('inf') if recent_period > 0 else 0
        cagr = 0
    
    # Trend strength score (composite metric)
    # Combines slope magnitude, statistical significance, and fit quality
    trend_strength = abs(slope_norm) * (1 - p_value) * r2_norm
    
    return {
        'slope_normalized': slope_norm,
        'r2_normalized': r2_norm,
        'p_value': p_value,
        't_statistic': t_stat,
        'percentage_growth': percentage_growth,
        'cagr': cagr,
        'trend_strength': trend_strength,
        'data_points': len(df),
        'recent_avg': recent_period,
        'early_avg': early_period,
        'is_significant': p_value < 0.05,
        'trend_direction': 'increasing' if slope_norm > 0 else 'decreasing'
    }

@st.cache_data(ttl=3600)
def get_words_in_frequency_range(min_occurrences=100, max_occurrences=3000):
    """Get all words within a specific frequency range"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get words that appear frequently across posts and comments
            cur.execute("""
                WITH word_frequencies AS (
                    SELECT 
                        unnest(string_to_array(
                            regexp_replace(
                                lower(COALESCE(markdown_content, '') || ' ' || COALESCE(title, '')), 
                                '[^a-zA-Z0-9\\s]', ' ', 'g'
                            ), 
                            ' '
                        )) as word,
                        COUNT(*) as freq
                    FROM fellowship_mvp 
                    WHERE markdown_content IS NOT NULL 
                    AND posted_at IS NOT NULL
                    GROUP BY 1
                    UNION ALL
                    SELECT 
                        unnest(string_to_array(
                            regexp_replace(
                                lower(COALESCE(markdown_content, '')), 
                                '[^a-zA-Z0-9\\s]', ' ', 'g'
                            ), 
                            ' '
                        )) as word,
                        COUNT(*) as freq
                    FROM fellowship_mvp_comments 
                    WHERE markdown_content IS NOT NULL
                    AND posted_at IS NOT NULL
                    GROUP BY 1
                ),
                word_totals AS (
                    SELECT word, SUM(freq) as total_freq
                    FROM word_frequencies
                    WHERE word != '' 
                    AND length(word) > 3  -- Only words longer than 3 characters
                    AND word NOT IN ('the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'may', 'she', 'use', 'way', 'will', 'with', 'have', 'this', 'that', 'they', 'from', 'would', 'there', 'their', 'what', 'been', 'said', 'each', 'which', 'more', 'very', 'when', 'come', 'much', 'were', 'here', 'than', 'like', 'time', 'make', 'about', 'after', 'first', 'well', 'many', 'some', 'then', 'them', 'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'while', 'could', 'other', 'before', 'should', 'through', 'between')
                    GROUP BY word
                    HAVING SUM(freq) BETWEEN %s AND %s
                )
                SELECT word, total_freq
                FROM word_totals
                ORDER BY total_freq DESC
            """, (min_occurrences, max_occurrences))
            
            results = cur.fetchall()
            return [{'word': row[0], 'frequency': row[1]} for row in results]

def analyze_all_words_in_range(min_occurrences=100, max_occurrences=3000, start_date=None, end_date=None):
    """Analyze trending words for all words in frequency range"""
    
    # Get words in frequency range
    all_words = get_words_in_frequency_range(min_occurrences, max_occurrences)
    word_list = [item['word'] for item in all_words]
    
    if not word_list:
        return pd.DataFrame()
    
    # Get monthly content counts
    monthly_counts = fetch_monthly_ea_content_counts()
    
    # Analyze trends for each word
    trend_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.info(f"🔍 Found **{len(word_list)} words** in frequency range {min_occurrences}-{max_occurrences}. Analyzing trends...")
    
    for i, word in enumerate(word_list):
        status_text.text(f'Analyzing trends for "{word}"... ({i+1}/{len(word_list)})')
        progress_bar.progress((i + 1) / len(word_list))
        
        try:
            # Get word occurrence data
            word_data_dict = fetch_word_occurrences_ea_cached(
                word, 
                start_date or date(2020, 1, 1), 
                end_date or date.today(), 
                []
            )
            
            word_data = word_data_dict.get(word, [])
            
            # Calculate trend metrics
            metrics = calculate_trend_metrics(word_data, monthly_counts)
            
            if metrics and metrics['data_points'] >= 3:
                trend_results.append({
                    'word': word,
                    'total_occurrences': next((item['frequency'] for item in all_words if item['word'] == word), 0),
                    **metrics
                })
        except Exception as e:
            # Only show errors for first few failures to avoid spam
            if len(trend_results) < 5:
                st.write(f"Error analyzing '{word}': {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(trend_results)

# Main Streamlit App
st.title("🎯 EA Forum: All Words Trend Analysis")

st.markdown("""
This analysis examines **every word** within a specific frequency range to find trending terms:
- **Comprehensive**: Analyzes all words meeting frequency criteria
- **No pre-filtering**: Discovers unexpected trending topics  
- **Statistical rigor**: Linear regression, p-values, trend strength scoring
- **Balanced scope**: Avoids both rare words (insufficient data) and super common words (uninteresting)
""")

# Parameters
col1, col2, col3 = st.columns(3)

with col1:
    min_occurrences = st.slider("Minimum word frequency", 50, 500, 100, 25)

with col2:
    max_occurrences = st.slider("Maximum word frequency", 500, 10000, 3000, 250)

with col3:
    date_range = st.date_input(
        "Analysis period",
        value=[date(2020, 1, 1), date.today()],
        format="YYYY-MM-DD",
        key="date_range"
    )

# Show estimated analysis scope
with st.expander("📊 Analysis Scope Preview"):
    if st.button("Count words in range", type="secondary"):
        with st.spinner("Counting words in frequency range..."):
            words_in_range = get_words_in_frequency_range(min_occurrences, max_occurrences)
            st.success(f"**{len(words_in_range)} words** found in range {min_occurrences}-{max_occurrences} occurrences")
            
            if len(words_in_range) > 0:
                st.write("**Sample words:**")
                sample_words = words_in_range[:20]  # Show first 20
                for i in range(0, len(sample_words), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        if i + j < len(sample_words):
                            word_info = sample_words[i + j]
                            col.write(f"• {word_info['word']} ({word_info['frequency']})")
                
                if len(words_in_range) > 20:
                    st.caption(f"... and {len(words_in_range) - 20} more words")
                    
                # Estimate time
                estimated_time = len(words_in_range) * 0.5  # rough estimate: 0.5 seconds per word
                st.info(f"⏱️ Estimated analysis time: ~{estimated_time/60:.1f} minutes for {len(words_in_range)} words")

st.info(f"🎯 **Analysis**: Words with {min_occurrences}-{max_occurrences} occurrences (balanced frequency range)")

if st.button("🔍 Analyze All Words in Range", type="primary"):
    start_date, end_date = date_range if len(date_range) == 2 else (date(2020, 1, 1), date.today())
    
    # Run the analysis
    with st.spinner("Running comprehensive word trend analysis..."):
        results_df = analyze_all_words_in_range(min_occurrences, max_occurrences, start_date, end_date)
    
    if not results_df.empty:
        st.success(f"✅ Analyzed **{len(results_df)} words** with sufficient trend data")
        
        # Show results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Top Trending Up", "📉 Top Trending Down", "📊 All Results", "📈 Analytics"])
        
        with tab1:
            st.subheader("🚀 Strongest Upward Trends")
            
            # Filter for significant positive trends
            trending_up = results_df[
                (results_df['trend_direction'] == 'increasing') & 
                (results_df['is_significant'] == True) &
                (results_df['percentage_growth'] > 0)
            ].sort_values('trend_strength', ascending=False).head(20)
            
            if not trending_up.empty:
                for idx, row in trending_up.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{row['word']}**")
                        st.caption(f"Total occurrences: {row['total_occurrences']:,}")
                    
                    with col2:
                        growth = row['percentage_growth']
                        if growth == float('inf'):
                            st.metric("Growth", "∞%", "🚀")
                        else:
                            st.metric("Growth", f"{growth:.1f}%", "📈")
                    
                    with col3:
                        st.metric("Trend Score", f"{row['trend_strength']:.3f}")
                    
                    with col4:
                        significance = "✅ Sig" if row['is_significant'] else "❌ NS"
                        st.metric("P-value", f"{row['p_value']:.3f}")
                        st.caption(significance)
                        
                    st.divider()
            else:
                st.info("No statistically significant upward trends found.")
        
        with tab2:
            st.subheader("📉 Strongest Downward Trends")
            
            trending_down = results_df[
                (results_df['trend_direction'] == 'decreasing') & 
                (results_df['is_significant'] == True) &
                (results_df['percentage_growth'] < 0)
            ].sort_values('trend_strength', ascending=False).head(20)
            
            if not trending_down.empty:
                for idx, row in trending_down.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{row['word']}**")
                        st.caption(f"Total occurrences: {row['total_occurrences']:,}")
                    
                    with col2:
                        growth = row['percentage_growth']
                        st.metric("Growth", f"{growth:.1f}%", "📉")
                    
                    with col3:
                        st.metric("Trend Score", f"{row['trend_strength']:.3f}")
                    
                    with col4:
                        significance = "✅ Sig" if row['is_significant'] else "❌ NS"
                        st.metric("P-value", f"{row['p_value']:.3f}")
                        st.caption(significance)
                        
                    st.divider()
            else:
                st.info("No statistically significant downward trends found.")
        
        with tab3:
            st.subheader("All Analysis Results")
            
            display_df = results_df[['word', 'total_occurrences', 'trend_direction', 'percentage_growth', 
                                   'slope_normalized', 'p_value', 'r2_normalized', 'trend_strength', 'is_significant']]
            
            display_df = display_df.round({
                'percentage_growth': 1,
                'slope_normalized': 4,
                'p_value': 3,
                'r2_normalized': 3,
                'trend_strength': 4
            })
            
            st.dataframe(
                display_df.sort_values('trend_strength', ascending=False),
                width='stretch',
                height=500
            )
        
        with tab4:
            st.subheader("📈 Trend Analytics")
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                significant_trends = results_df[results_df['is_significant'] == True]
                st.metric("Significant Trends", len(significant_trends))
            
            with col2:
                increasing_trends = results_df[results_df['trend_direction'] == 'increasing']
                st.metric("Increasing Trends", len(increasing_trends))
            
            with col3:
                decreasing_trends = results_df[results_df['trend_direction'] == 'decreasing']
                st.metric("Decreasing Trends", len(decreasing_trends))
            
            with col4:
                avg_strength = results_df['trend_strength'].mean()
                st.metric("Avg Trend Strength", f"{avg_strength:.3f}")
            
            # Visualization
            if len(results_df) > 0:
                st.subheader("📊 Trend Strength vs Word Frequency")
                
                fig = px.scatter(
                    results_df,
                    x='total_occurrences',
                    y='trend_strength',
                    hover_name='word',
                    color='trend_direction',
                    size='r2_normalized',
                    title="Word Frequency vs Trend Strength",
                    labels={
                        'total_occurrences': 'Total Word Occurrences',
                        'trend_strength': 'Trend Strength Score',
                        'trend_direction': 'Trend Direction'
                    }
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No words found with sufficient data for analysis. Try adjusting the frequency range.")

# Methodology explanation
with st.expander("📚 Comprehensive Analysis Methodology"):
    st.markdown("""
    ### What Makes This Different:
    
    1. **No Pre-filtering by Domain**
       - Analyzes ALL words in frequency range
       - Discovers unexpected trending topics
       - Unbiased approach to trend detection
    
    2. **Balanced Frequency Range**
       - Minimum threshold: Ensures sufficient data for statistical analysis
       - Maximum threshold: Avoids overly common words with boring trends
       - Sweet spot: Captures meaningful but not ubiquitous terms
    
    3. **Statistical Rigor**
       - Linear regression on normalized word counts
       - P-value testing for trend significance  
       - R² scoring for trend fit quality
       - Composite trend strength metric
    
    4. **Comprehensive Coverage**
       - Posts AND comments analyzed
       - Time series data normalized by monthly activity
       - Growth rate calculations (percentage change)
    
    5. **Quality Filtering**
       - Excludes very short words (≤3 characters)
       - Removes common English stop words
       - Focuses on content words with semantic meaning
    """)

if __name__ == "__main__":
    pass