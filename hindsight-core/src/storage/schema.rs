/// SQLite schema initialization
pub fn create_tables() -> &'static str {
    "
    CREATE TABLE IF NOT EXISTS banks (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        disposition TEXT NOT NULL DEFAULT '{\"skepticism\":3,\"literalism\":3,\"empathy\":3}',
        background_context TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        bank_id TEXT NOT NULL REFERENCES banks(id),
        content TEXT NOT NULL,
        content_type TEXT DEFAULT 'text/plain',
        title TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS memory_units (
        id TEXT PRIMARY KEY,
        bank_id TEXT NOT NULL REFERENCES banks(id),
        fact_type TEXT NOT NULL DEFAULT 'experience',
        content TEXT NOT NULL,
        content_search TEXT NOT NULL DEFAULT '',
        summary TEXT,
        summary_search TEXT NOT NULL DEFAULT '',
        source_document_id TEXT REFERENCES documents(id),
        confidence REAL DEFAULT 1.0,
        tags TEXT DEFAULT '[]',
        metadata TEXT DEFAULT '{}',
        occurred_start TEXT,
        occurred_end TEXT,
        mentioned_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,
        memory_unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
        embedding BLOB NOT NULL,
        model TEXT NOT NULL,
        dimensions INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS memory_links (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES memory_units(id),
        target_id TEXT NOT NULL REFERENCES memory_units(id),
        link_type TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        bank_id TEXT NOT NULL REFERENCES banks(id),
        name TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        labels TEXT DEFAULT '[]',
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS unit_entities (
        memory_unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
        entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        PRIMARY KEY (memory_unit_id, entity_id)
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_memory_units_bank ON memory_units(bank_id, fact_type);
    CREATE INDEX IF NOT EXISTS idx_memory_units_temporal ON memory_units(bank_id, occurred_start, occurred_end);
    CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_id, link_type);
    CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_id, link_type);
    CREATE INDEX IF NOT EXISTS idx_entities_bank ON entities(bank_id, name);
    CREATE INDEX IF NOT EXISTS idx_embeddings_memory ON embeddings(memory_unit_id);

    -- FTS5 full-text search (indexes pre-segmented Chinese text)
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_units_fts USING fts5(
        content_search,
        summary_search,
        content='memory_units',
        content_rowid='rowid'
    );

    -- FTS sync triggers (use pre-segmented columns)
    CREATE TRIGGER IF NOT EXISTS memory_units_ai AFTER INSERT ON memory_units BEGIN
        INSERT INTO memory_units_fts(rowid, content_search, summary_search) VALUES (new.rowid, new.content_search, new.summary_search);
    END;

    CREATE TRIGGER IF NOT EXISTS memory_units_ad AFTER DELETE ON memory_units BEGIN
        INSERT INTO memory_units_fts(memory_units_fts, rowid, content_search, summary_search) VALUES('delete', old.rowid, old.content_search, old.summary_search);
    END;

    CREATE TRIGGER IF NOT EXISTS memory_units_au AFTER UPDATE ON memory_units BEGIN
        INSERT INTO memory_units_fts(memory_units_fts, rowid, content_search, summary_search) VALUES('delete', old.rowid, old.content_search, old.summary_search);
        INSERT INTO memory_units_fts(rowid, content_search, summary_search) VALUES (new.rowid, new.content_search, new.summary_search);
    END;
    "
}
