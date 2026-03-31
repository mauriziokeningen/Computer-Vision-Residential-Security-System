-- ==============================================================================
-- 1. EXTENSIONES DEL SISTEMA
-- ==============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ==============================================================================
-- 2. TABLAS PADRE (Entidades Independientes)
-- ==============================================================================

-- Catálogo Único de Identidades (Residentes, Visitantes, Staff)
CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(100) NOT NULL,
    person_type VARCHAR(50) NOT NULL,       -- Ej: 'RESIDENT', 'VISITOR', 'STAFF'
    building VARCHAR(100),                  -- Nullable (un visitante puede no vivir ahí)
    apartment VARCHAR(100),                 -- Nullable
    phone VARCHAR(20) UNIQUE,               -- Nullable (quizá no tenemos el teléfono del visitante)
    email VARCHAR(100) UNIQUE,              -- Nullable
    valid_from TIMESTAMP,                   -- Para accesos temporales (Visitantes)
    valid_until TIMESTAMP,                  -- Para accesos temporales (Visitantes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_embedding VECTOR(512)
);

-- Catálogo de Infraestructura (Cámaras físicas)
CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location VARCHAR(250) UNIQUE NOT NULL,
    ip_address VARCHAR(250) UNIQUE NOT NULL, 
    status VARCHAR(100) NOT NULL
);

-- El "Paraguas" del Incidente (La verdad inmutable)
CREATE TABLE incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    incident_metadata JSONB NOT NULL
);

-- ==============================================================================
-- 3. TABLAS HIJO (Relaciones, Evidencia y Frontend)
-- ==============================================================================

-- La "Caja de Evidencias": Rastreo temporal y espacial
CREATE TABLE incident_timeline (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID REFERENCES incidents(id),
    camera_id UUID REFERENCES cameras(id),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    video_url VARCHAR(255) NOT NULL
);

-- La "Lista de Sospechosos": Relación Muchos a Muchos actualizada a 'persons'
CREATE TABLE incidents_involved (
    person_id UUID REFERENCES persons(id),   -- Actualizado para apuntar a 'persons'
    incident_id UUID REFERENCES incidents(id),
    PRIMARY KEY (person_id, incident_id)
);

-- Bandeja de Notificaciones para el Frontend (React)
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID REFERENCES incidents(id),
    message TEXT NOT NULL,                  -- Ej: "Arma detectada en Lobby"
    status VARCHAR(50) DEFAULT 'UNREAD',    -- 'UNREAD', 'ACKNOWLEDGED', 'RESOLVED'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP                   -- Se llena cuando el guardia atiende la alerta
);