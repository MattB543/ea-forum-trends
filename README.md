# 📈 EA Forum Trends Analysis

**Interactive trend analysis and keyword search for EA Forum data** - Discover what's trending, explore content, and analyze conversation patterns in the Effective Altruism community.

## 🚀 Live Applications

- **Interactive Trends**: [Coming Soon] - Search keywords, visualize trends, discover content
- **Comprehensive Analysis**: [Coming Soon] - Statistical analysis of all words in frequency ranges

## ✨ Features

### 📊 Interactive Trends App (`ea_forum_trends_sync.py`)
- **Multi-keyword search** with autocomplete suggestions
- **Interactive trend visualization** with time-series charts  
- **Content discovery** - click and drag on charts to filter results
- **Author filtering** for personalized analysis
- **Normalized metrics** accounting for forum activity changes
- **Real-time content highlighting** of search terms

### 🔍 Comprehensive Analysis App (`trending_words_frequency_range.py`)  
- **Exhaustive word analysis** - analyze ALL words within frequency ranges
- **Statistical rigor** with linear regression, p-values, and trend scoring
- **Unbiased discovery** - find unexpected trending topics without pre-filtering
- **Advanced analytics** with trend strength scoring and growth metrics
- **Frequency range optimization** - balance between rare words (insufficient data) and common words (uninteresting)

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database with EA Forum data
- Streamlit account (for deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/MattB543/ea-forum-trends.git
   cd ea-forum-trends
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database connection string
   ```

4. **Set up database functions**
   ```bash
   # Run the SQL setup file in your PostgreSQL database
   psql -d your_database -f setup_ea_forum_functions.sql
   ```

5. **Run the applications locally**
   ```bash
   # Interactive trends app
   streamlit run ea_forum_trends_sync.py --server.port 8501

   # Comprehensive analysis app  
   streamlit run trending_words_frequency_range.py --server.port 8502
   ```

## 🌐 Deploying to Streamlit Community Cloud

This repository is designed to deploy **two separate Streamlit applications** from the same repo:

### Application 1: Interactive Trends
- **Entry point**: `ea_forum_trends_sync.py` 
- **Purpose**: Real-time keyword search and trend visualization
- **Best for**: Exploring specific topics and discovering related content

### Application 2: Comprehensive Analysis  
- **Entry point**: `trending_words_frequency_range.py`
- **Purpose**: Statistical analysis of word trends across frequency ranges
- **Best for**: Discovering unexpected trends and comprehensive trend analysis

### Deployment Steps

1. **Fork or import this repo** to your GitHub account

2. **Deploy App 1 (Interactive Trends)**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `MattB543/ea-forum-trends`
   - **Main file path**: `ea_forum_trends_sync.py`
   - Add secrets: `DATABASE_URL = your_connection_string`
   - Deploy!

3. **Deploy App 2 (Comprehensive Analysis)**:
   - Click "New app" again
   - Select the same repository: `MattB543/ea-forum-trends` 
   - **Main file path**: `trending_words_frequency_range.py`
   - Add the same secrets: `DATABASE_URL = your_connection_string`
   - Deploy!

Each app will get its own unique URL (e.g., `yourname-ea-trends-1.streamlit.app` and `yourname-ea-trends-2.streamlit.app`).

## 📊 Database Schema

The applications expect PostgreSQL tables with EA Forum data:

### Required Tables
- `fellowship_mvp` - Posts table with columns:
  - `post_id`, `title`, `markdown_content`, `author_id`, `author_display_name`, `posted_at`, `page_url`, `base_score`
  
- `fellowship_mvp_comments` - Comments table with columns:
  - `comment_id`, `post_id`, `markdown_content`, `author_id`, `author_display_name`, `posted_at`, `base_score`

### Database Functions
The `setup_ea_forum_functions.sql` file creates necessary PostgreSQL functions:
- `search_ea_content()` - Full-text search across posts and comments
- `word_occurrences_ea()` - Time-series word occurrence tracking  
- `get_monthly_ea_content_counts()` - Content volume normalization
- `get_ea_authors()` - Author filtering functionality
- `get_ea_global_stats()` - Forum statistics

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL` - PostgreSQL connection string (required)

### Streamlit Configuration  
The `.streamlit/config.toml` file includes optimized settings for both applications:
- Performance optimizations
- Theme customization  
- Caching configuration

## 📈 Analysis Methodology

### Interactive Trends App
- **Full-text search** using PostgreSQL's `to_tsvector` and `plainto_tsquery`
- **Time-series visualization** with Plotly interactive charts
- **Content normalization** per 1000 posts/comments to account for forum growth
- **Real-time filtering** based on chart selections

### Comprehensive Analysis App  
- **Statistical trend detection** using linear regression
- **Significance testing** with p-values and t-statistics  
- **Trend strength scoring**: `|slope| × (1 - p_value) × R²`
- **Growth rate calculations** comparing recent vs. early periods
- **Frequency range optimization** to balance statistical power vs. meaningfulness

## 🎯 Use Cases

### Researchers & Analysts
- Track emerging topics in EA discourse
- Analyze conversation trends over time
- Discover unexpected trending concepts
- Filter by specific authors or time periods

### Community Members
- Explore what the community is discussing
- Find related content on topics of interest
- Understand how conversations evolve
- Discover new areas of EA research

### Content Creators
- Identify trending topics for content creation
- Understand audience interests over time
- Find related discussions and build on them
- Track reception of different EA concepts

## 🔍 Advanced Features

### Statistical Analysis
- **Linear regression** trend fitting
- **R² scoring** for trend quality assessment
- **P-value testing** for statistical significance
- **Mann-Kendall tests** for monotonic trends (in some versions)

### Performance Optimizations
- **Caching** with 1-hour TTL for database queries
- **Batch processing** for multiple word analysis
- **Progressive loading** with progress bars
- **Efficient database indexing** with GIN indexes for full-text search

### Visualization
- **Interactive Plotly charts** with zoom, pan, and selection
- **Multi-series line plots** for keyword comparison  
- **Scatter plots** for trend strength vs. frequency analysis
- **Responsive design** optimized for various screen sizes

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Additional statistical methods** (ARIMA, seasonal decomposition, etc.)
- **Enhanced visualizations** (heatmaps, network graphs, etc.)
- **Export functionality** (CSV, PDF reports, etc.)  
- **Advanced filtering** (by karma, post length, etc.)
- **Semantic analysis** (topic modeling, sentiment analysis, etc.)

## 📜 License

MIT License - feel free to adapt for your own community analysis needs!

## 🙏 Acknowledgments

- Built for the **Effective Altruism Forum** community
- Inspired by trend analysis tools and EA research priorities
- Uses PostgreSQL full-text search and Python data science stack
- Deployed on **Streamlit Community Cloud** for easy sharing

---

**Questions?** Open an issue or contribute to make EA Forum trend analysis even better! 🚀