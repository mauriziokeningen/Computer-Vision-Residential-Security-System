-- ==============================================================================
-- 1. SYSTEM EXTENSIONS
-- ==============================================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ==============================================================================
-- 2. CORE / PARENT TABLES (Independent Entities)
-- ==============================================================================

CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(100) NOT NULL,
    person_type VARCHAR(50) NOT NULL,
    building VARCHAR(100),
    apartment VARCHAR(100),
    phone VARCHAR(20) UNIQUE,
    email VARCHAR(100) UNIQUE,
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_embedding VECTOR(512)
);

CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location VARCHAR(250) UNIQUE NOT NULL,
    ip_address VARCHAR(250) UNIQUE NOT NULL,
    status VARCHAR(100) NOT NULL
);

CREATE TABLE incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    incident_metadata JSONB NOT NULL
);

-- ==============================================================================
-- 3. DEPENDENT / CHILD TABLES (Relationships, Evidence, and System Actions)
-- ==============================================================================

CREATE TABLE incident_timeline (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID REFERENCES incidents(id),
    camera_id UUID REFERENCES cameras(id),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    video_url VARCHAR(255) NOT NULL
);

CREATE TABLE incidents_involved (
    person_id UUID REFERENCES persons(id),
    incident_id UUID REFERENCES incidents(id),
    PRIMARY KEY (person_id, incident_id)
);

CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID REFERENCES incidents(id),
    message TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'UNREAD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- ==============================================================================
-- 4. HIGH-PERFORMANCE VECTOR INDEXING (HNSW)
-- ==============================================================================

-- Converts the computational complexity of biometric verification from a linear 
-- sequential scan O(N) into an Approximate Nearest Neighbor logarithmic search O(log N).
CREATE INDEX IF NOT EXISTS idx_persons_face_embedding_hnsw
ON persons
USING hnsw (face_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);