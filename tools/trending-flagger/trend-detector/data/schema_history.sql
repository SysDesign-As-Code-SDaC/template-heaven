CREATE TABLE repository_history (
    id SERIAL PRIMARY KEY,
    repository_url VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    stars INTEGER NOT NULL,
    forks INTEGER NOT NULL,
    watchers INTEGER NOT NULL,
    issues INTEGER NOT NULL,
    pull_requests INTEGER NOT NULL,
    commits INTEGER NOT NULL,
    contributors INTEGER NOT NULL
);