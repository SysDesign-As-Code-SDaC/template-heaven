-- Template Heaven Trend Detection Database Schema
-- This schema supports comprehensive trend monitoring and template management

-- Repositories table - stores basic repository information
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    github_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    description TEXT,
    url VARCHAR(500) NOT NULL,
    clone_url VARCHAR(500),
    language VARCHAR(100),
    topics TEXT[],
    size INTEGER,
    license VARCHAR(100),
    default_branch VARCHAR(100) DEFAULT 'main',
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    last_checked_at TIMESTAMP,
    is_template BOOLEAN DEFAULT FALSE,
    template_type VARCHAR(50),
    stack_category VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE
);

-- Repository metrics table - stores time-series metrics for repositories
CREATE TABLE repository_metrics (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    stars INTEGER NOT NULL DEFAULT 0,
    forks INTEGER NOT NULL DEFAULT 0,
    watchers INTEGER NOT NULL DEFAULT 0,
    open_issues INTEGER NOT NULL DEFAULT 0,
    open_pull_requests INTEGER NOT NULL DEFAULT 0,
    total_commits INTEGER NOT NULL DEFAULT 0,
    contributors INTEGER NOT NULL DEFAULT 0,
    recent_commits INTEGER NOT NULL DEFAULT 0, -- commits in last 30 days
    recent_issues INTEGER NOT NULL DEFAULT 0,  -- issues in last 30 days
    recent_pull_requests INTEGER NOT NULL DEFAULT 0, -- PRs in last 30 days
    star_growth_rate REAL DEFAULT 0.0,
    fork_growth_rate REAL DEFAULT 0.0,
    activity_score REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.0,
    trend_score REAL DEFAULT 0.0,
    priority_score REAL DEFAULT 0.0,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(repository_id, recorded_at)
);

-- Trend alerts table - stores alerts for trending repositories
CREATE TABLE trend_alerts (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- 'high_stars', 'rapid_growth', 'early_trend', 'template_candidate'
    trend_level VARCHAR(50) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    trend_score REAL NOT NULL,
    priority_score REAL NOT NULL,
    confidence_score REAL DEFAULT 0.0,
    trend_reasons TEXT[],
    stack_relevance REAL DEFAULT 0.0,
    template_quality_score REAL DEFAULT 0.0,
    human_review_required BOOLEAN NOT NULL DEFAULT TRUE,
    review_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'ignored'
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP,
    review_notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Stack configurations table - stores stack-specific trend detection settings
CREATE TABLE stack_configurations (
    id SERIAL PRIMARY KEY,
    stack_name VARCHAR(100) UNIQUE NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    keywords TEXT[],
    characteristics JSONB,
    thresholds JSONB,
    auto_sync_enabled BOOLEAN DEFAULT FALSE,
    require_approval BOOLEAN DEFAULT TRUE,
    notification_channels TEXT[],
    priority_threshold REAL DEFAULT 0.7,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Template candidates table - repositories identified as potential templates
CREATE TABLE template_candidates (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    stack_category VARCHAR(100),
    template_name VARCHAR(255),
    template_description TEXT,
    template_quality_score REAL DEFAULT 0.0,
    documentation_score REAL DEFAULT 0.0,
    code_quality_score REAL DEFAULT 0.0,
    maintenance_score REAL DEFAULT 0.0,
    popularity_score REAL DEFAULT 0.0,
    overall_score REAL DEFAULT 0.0,
    is_approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP,
    rejection_reason TEXT,
    sync_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'synced', 'failed', 'ignored'
    sync_attempts INTEGER DEFAULT 0,
    last_sync_attempt TIMESTAMP,
    sync_error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Sync history table - tracks template synchronization history
CREATE TABLE sync_history (
    id SERIAL PRIMARY KEY,
    template_candidate_id INTEGER REFERENCES template_candidates(id) ON DELETE CASCADE,
    sync_type VARCHAR(50) NOT NULL, -- 'initial', 'update', 'force'
    sync_status VARCHAR(50) NOT NULL, -- 'success', 'failed', 'partial'
    source_url VARCHAR(500) NOT NULL,
    target_path VARCHAR(500) NOT NULL,
    files_synced INTEGER DEFAULT 0,
    files_updated INTEGER DEFAULT 0,
    files_failed INTEGER DEFAULT 0,
    sync_duration INTERVAL,
    error_message TEXT,
    synced_by VARCHAR(255),
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Notifications table - tracks sent notifications
CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    notification_type VARCHAR(50) NOT NULL, -- 'trend_alert', 'sync_complete', 'review_required'
    channel VARCHAR(50) NOT NULL, -- 'slack', 'email', 'github_issue'
    recipient VARCHAR(255),
    subject VARCHAR(500),
    message TEXT,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'sent', 'failed'
    sent_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX idx_repositories_github_id ON repositories(github_id);
CREATE INDEX idx_repositories_full_name ON repositories(full_name);
CREATE INDEX idx_repositories_language ON repositories(language);
CREATE INDEX idx_repositories_stack_category ON repositories(stack_category);
CREATE INDEX idx_repositories_is_template ON repositories(is_template);
CREATE INDEX idx_repositories_updated_at ON repositories(updated_at);

CREATE INDEX idx_repository_metrics_repository_id ON repository_metrics(repository_id);
CREATE INDEX idx_repository_metrics_recorded_at ON repository_metrics(recorded_at);
CREATE INDEX idx_repository_metrics_trend_score ON repository_metrics(trend_score);
CREATE INDEX idx_repository_metrics_priority_score ON repository_metrics(priority_score);

CREATE INDEX idx_trend_alerts_repository_id ON trend_alerts(repository_id);
CREATE INDEX idx_trend_alerts_alert_type ON trend_alerts(alert_type);
CREATE INDEX idx_trend_alerts_trend_level ON trend_alerts(trend_level);
CREATE INDEX idx_trend_alerts_created_at ON trend_alerts(created_at);
CREATE INDEX idx_trend_alerts_review_status ON trend_alerts(review_status);

CREATE INDEX idx_template_candidates_repository_id ON template_candidates(repository_id);
CREATE INDEX idx_template_candidates_stack_category ON template_candidates(stack_category);
CREATE INDEX idx_template_candidates_is_approved ON template_candidates(is_approved);
CREATE INDEX idx_template_candidates_sync_status ON template_candidates(sync_status);

CREATE INDEX idx_sync_history_template_candidate_id ON sync_history(template_candidate_id);
CREATE INDEX idx_sync_history_synced_at ON sync_history(synced_at);

CREATE INDEX idx_notifications_type ON notifications(notification_type);
CREATE INDEX idx_notifications_channel ON notifications(channel);
CREATE INDEX idx_notifications_status ON notifications(status);
CREATE INDEX idx_notifications_created_at ON notifications(created_at);

-- Views for common queries
CREATE VIEW trending_repositories AS
SELECT 
    r.id,
    r.full_name,
    r.description,
    r.url,
    r.language,
    r.topics,
    r.stack_category,
    rm.stars,
    rm.forks,
    rm.trend_score,
    rm.priority_score,
    rm.recorded_at
FROM repositories r
JOIN repository_metrics rm ON r.id = rm.repository_id
WHERE rm.trend_score > 0.5
ORDER BY rm.trend_score DESC, rm.priority_score DESC;

CREATE VIEW high_priority_alerts AS
SELECT 
    ta.id,
    r.full_name,
    r.url,
    ta.alert_type,
    ta.trend_level,
    ta.trend_score,
    ta.priority_score,
    ta.trend_reasons,
    ta.created_at
FROM trend_alerts ta
JOIN repositories r ON ta.repository_id = r.id
WHERE ta.trend_level IN ('high', 'critical')
AND ta.review_status = 'pending'
ORDER BY ta.priority_score DESC, ta.created_at DESC;

CREATE VIEW approved_templates AS
SELECT 
    tc.id,
    r.full_name,
    r.url,
    tc.stack_category,
    tc.template_name,
    tc.overall_score,
    tc.sync_status,
    tc.approved_at
FROM template_candidates tc
JOIN repositories r ON tc.repository_id = r.id
WHERE tc.is_approved = TRUE
ORDER BY tc.overall_score DESC, tc.approved_at DESC;

-- Functions for common operations
CREATE OR REPLACE FUNCTION update_repository_metrics(
    p_repository_id INTEGER,
    p_stars INTEGER,
    p_forks INTEGER,
    p_watchers INTEGER,
    p_open_issues INTEGER,
    p_open_pull_requests INTEGER,
    p_total_commits INTEGER,
    p_contributors INTEGER,
    p_recent_commits INTEGER,
    p_recent_issues INTEGER,
    p_recent_pull_requests INTEGER
) RETURNS VOID AS $$
BEGIN
    INSERT INTO repository_metrics (
        repository_id, stars, forks, watchers, open_issues, open_pull_requests,
        total_commits, contributors, recent_commits, recent_issues, recent_pull_requests,
        recorded_at
    ) VALUES (
        p_repository_id, p_stars, p_forks, p_watchers, p_open_issues, p_open_pull_requests,
        p_total_commits, p_contributors, p_recent_commits, p_recent_issues, p_recent_pull_requests,
        CURRENT_TIMESTAMP
    );
    
    -- Update the repository's last_checked_at timestamp
    UPDATE repositories 
    SET last_checked_at = CURRENT_TIMESTAMP 
    WHERE id = p_repository_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_repositories_updated_at 
    BEFORE UPDATE ON repositories 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trend_alerts_updated_at 
    BEFORE UPDATE ON trend_alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_template_candidates_updated_at 
    BEFORE UPDATE ON template_candidates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stack_configurations_updated_at 
    BEFORE UPDATE ON stack_configurations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();