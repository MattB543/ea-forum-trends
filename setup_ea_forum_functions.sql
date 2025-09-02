-- Database functions for EA Forum trends analysis
-- Adapted for fellowship_mvp and fellowship_mvp_comments tables

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS search_ea_content(TEXT, DATE, DATE, INTEGER);
DROP FUNCTION IF EXISTS word_occurrences_ea(TEXT, TEXT[]);
DROP FUNCTION IF EXISTS get_monthly_ea_content_counts();
DROP FUNCTION IF EXISTS get_ea_authors();
DROP FUNCTION IF EXISTS get_ea_global_stats();

-- Create full-text search indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_fellowship_mvp_content_fts 
ON fellowship_mvp USING gin(to_tsvector('english', COALESCE(markdown_content, '') || ' ' || COALESCE(title, '')));

CREATE INDEX IF NOT EXISTS idx_fellowship_mvp_comments_content_fts 
ON fellowship_mvp_comments USING gin(to_tsvector('english', COALESCE(markdown_content, '')));

-- Search posts and comments function
CREATE OR REPLACE FUNCTION search_ea_content(
    search_query TEXT,
    since_date DATE,
    until_date DATE,
    limit_ INTEGER DEFAULT 100
) RETURNS TABLE (
    content_id TEXT,
    content_type TEXT,
    title TEXT,
    content TEXT,
    author_display_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    url TEXT,
    score INTEGER,
    mention_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    (SELECT 
        p.post_id::TEXT as content_id,
        'post'::TEXT as content_type,
        COALESCE(p.title, '')::TEXT as title,
        COALESCE(p.markdown_content, '')::TEXT as content,
        COALESCE(p.author_display_name, '')::TEXT as author_display_name,
        p.posted_at as created_at,
        COALESCE(p.page_url, '')::TEXT as url,
        COALESCE(p.base_score, 0) as score,
        -- Count actual mentions of the search term in title + content (substring-based, case-insensitive)
        CASE 
            WHEN search_query IS NULL OR length(trim(search_query)) = 0 THEN 0
            ELSE (length(lower(COALESCE(p.title, '') || ' ' || COALESCE(p.markdown_content, ''))) - 
                  length(replace(lower(COALESCE(p.title, '') || ' ' || COALESCE(p.markdown_content, '')), lower(trim(search_query)), '')))::INTEGER / length(trim(search_query))
        END as mention_count
    FROM fellowship_mvp p
    WHERE to_tsvector('english', COALESCE(p.markdown_content, '') || ' ' || COALESCE(p.title, '')) @@ plainto_tsquery('english', search_query)
    AND p.posted_at::date BETWEEN since_date AND until_date
    AND p.posted_at IS NOT NULL
    ORDER BY p.posted_at DESC
    LIMIT limit_)
    UNION ALL
    (SELECT 
        c.comment_id::TEXT as content_id,
        'comment'::TEXT as content_type,
        COALESCE(p.title, '')::TEXT as title,
        COALESCE(c.markdown_content, '')::TEXT as content,
        COALESCE(c.author_display_name, '')::TEXT as author_display_name,
        c.posted_at as created_at,
        (COALESCE(p.page_url, '') || '#comment-' || c.comment_id)::TEXT as url,
        COALESCE(c.base_score, 0) as score,
        -- Count actual mentions of the search term in comment content (substring-based, case-insensitive)
        CASE 
            WHEN search_query IS NULL OR length(trim(search_query)) = 0 THEN 0
            ELSE (length(lower(COALESCE(c.markdown_content, ''))) - 
                  length(replace(lower(COALESCE(c.markdown_content, '')), lower(trim(search_query)), '')))::INTEGER / length(trim(search_query))
        END as mention_count
    FROM fellowship_mvp_comments c
    JOIN fellowship_mvp p ON c.post_id = p.post_id
    WHERE to_tsvector('english', COALESCE(c.markdown_content, '')) @@ plainto_tsquery('english', search_query)
    AND c.posted_at::date BETWEEN since_date AND until_date
    AND c.posted_at IS NOT NULL
    ORDER BY c.posted_at DESC
    LIMIT limit_);
END;
$$ LANGUAGE plpgsql;

-- Word occurrences over time function
CREATE OR REPLACE FUNCTION word_occurrences_ea(
    search_word TEXT,
    author_ids TEXT[] DEFAULT NULL
) RETURNS TABLE (
    month TEXT,
    word_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH monthly_counts AS (
        SELECT 
            to_char(posted_at, 'YYYY-MM') as month_key,
            COUNT(*) as count
        FROM (
            SELECT posted_at FROM fellowship_mvp 
            WHERE (author_ids IS NULL OR author_id = ANY(author_ids))
            AND to_tsvector('english', COALESCE(markdown_content, '') || ' ' || COALESCE(title, '')) @@ plainto_tsquery('english', search_word)
            AND posted_at IS NOT NULL
            UNION ALL
            SELECT posted_at FROM fellowship_mvp_comments 
            WHERE (author_ids IS NULL OR author_id = ANY(author_ids))
            AND to_tsvector('english', COALESCE(markdown_content, '')) @@ plainto_tsquery('english', search_word)
            AND posted_at IS NOT NULL
        ) combined
        GROUP BY to_char(posted_at, 'YYYY-MM')
    )
    SELECT month_key as month, count as word_count
    FROM monthly_counts
    ORDER BY month_key;
END;
$$ LANGUAGE plpgsql;

-- Monthly content counts for normalization
CREATE OR REPLACE FUNCTION get_monthly_ea_content_counts()
RETURNS TABLE (
    month TIMESTAMP WITH TIME ZONE,
    content_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH monthly_totals AS (
        SELECT 
            date_trunc('month', posted_at) as month_key,
            COUNT(*) as count
        FROM (
            SELECT posted_at FROM fellowship_mvp WHERE posted_at IS NOT NULL
            UNION ALL
            SELECT posted_at FROM fellowship_mvp_comments WHERE posted_at IS NOT NULL
        ) combined
        GROUP BY date_trunc('month', posted_at)
    )
    SELECT month_key as month, count as content_count
    FROM monthly_totals
    ORDER BY month_key;
END;
$$ LANGUAGE plpgsql;

-- Get unique authors for filtering
CREATE OR REPLACE FUNCTION get_ea_authors()
RETURNS TABLE (
    author_id TEXT,
    author_display_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH unique_authors AS (
        SELECT DISTINCT 
            COALESCE(p.author_id, '')::TEXT as author_id,
            COALESCE(p.author_display_name, 'Unknown')::TEXT as author_display_name
        FROM fellowship_mvp p
        WHERE p.author_id IS NOT NULL AND p.author_display_name IS NOT NULL
        UNION
        SELECT DISTINCT 
            COALESCE(c.author_id, '')::TEXT as author_id,
            COALESCE(c.author_display_name, 'Unknown')::TEXT as author_display_name
        FROM fellowship_mvp_comments c
        WHERE c.author_id IS NOT NULL AND c.author_display_name IS NOT NULL
    )
    SELECT u.author_id, u.author_display_name
    FROM unique_authors u
    WHERE u.author_display_name != 'Unknown'
    ORDER BY u.author_display_name;
END;
$$ LANGUAGE plpgsql;

-- Get global EA Forum statistics
CREATE OR REPLACE FUNCTION get_ea_global_stats()
RETURNS TABLE (
    total_posts BIGINT,
    total_comments BIGINT,
    total_authors BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM fellowship_mvp WHERE posted_at IS NOT NULL) as total_posts,
        (SELECT COUNT(*) FROM fellowship_mvp_comments WHERE posted_at IS NOT NULL) as total_comments,
        (SELECT COUNT(DISTINCT COALESCE(author_id, '')) 
         FROM (
             SELECT author_id FROM fellowship_mvp WHERE author_id IS NOT NULL
             UNION 
             SELECT author_id FROM fellowship_mvp_comments WHERE author_id IS NOT NULL
         ) combined) as total_authors;
END;
$$ LANGUAGE plpgsql;