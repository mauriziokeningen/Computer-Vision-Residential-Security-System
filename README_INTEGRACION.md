# 🚀 Guía de arranque — Computer Vision Residential Security System

## Requisitos previos

| Herramienta | Versión mínima | Instalación |
|---|---|---|
| Python | 3.10+ | python.org |
| Node.js | 18+ | nodejs.org |
| Docker Desktop | cualquiera | docker.com |

---

## Paso 1 — Levantar la base de datos y MinIO con Docker

```bash
# Desde la raíz del proyecto
docker-compose up -d
```

Esto levanta:
- **PostgreSQL + pgvector** en `localhost:5432`
- **MinIO** (object storage) en `localhost:9000` / consola en `localhost:9001`

Verifica que ambos contenedores estén corriendo:
```bash
docker ps
```

---

## Paso 1.1 — Si ya tenías un contenedor anterior, reinicia la base de datos

Si antes corriste otro backend o una versión vieja del proyecto, el contenedor de PostgreSQL puede haberse quedado con un esquema anterior y entonces las rutas de `alerts`, `incidents` y `persons` responden con **HTTP 500**.

Ejecuta esto antes de volver a levantar todo:

```bash
docker-compose down -v
```

Luego vuelve a correr:

```bash
docker-compose up -d
```

---

## Paso 2 — Instalar dependencias del backend

```bash
cd backend
pip install -r requirements.txt
```

> Si usas un entorno virtual (recomendado):
> ```bash
> python -m venv venv
> source venv/bin/activate        # Mac/Linux
> venv\Scripts\activate           # Windows
> pip install -r requirements.txt
> ```

---

## Paso 3 — Arrancar el backend (API REST + WebSocket)

```bash
# Desde la carpeta backend/
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Verifica que funcione abriendo: http://localhost:8000/health

Swagger UI disponible en: http://localhost:8000/api/docs

---

## Paso 4 — Instalar dependencias del frontend

```bash
# En otra terminal, desde la raíz del proyecto
cd frontend
npm install
```

---

## Paso 5 — Arrancar el frontend

```bash
# Dentro de frontend/
npm run dev
```

Abre el panel en: **http://localhost:5173**

---

## Estructura de puertos

| Servicio | Puerto | URL |
|---|---|---|
| Frontend (Vite + React) | 5173 | http://localhost:5173 |
| Backend (FastAPI) | 8000 | http://localhost:8000 |
| Swagger UI | 8000 | http://localhost:8000/api/docs |
| PostgreSQL | 5432 | — |
| MinIO S3 API | 9000 | http://localhost:9000 |
| MinIO Consola Web | 9001 | http://localhost:9001 |

---

## Cómo probar la integración sin los módulos de IA

1. Abre el panel en http://localhost:5173
2. Ve a la sección **Incidentes**
3. Usa el panel **"Simular evento"** para crear incidentes de prueba:
   - **Persona desconocida** → módulo `face`
   - **Arma** → módulo `weapons`
   - **Agresión** → módulo `pose`
4. Las alertas aparecerán automáticamente en **Alertas** y **Dashboard** vía WebSocket
5. Puedes cambiar el estado de cada alerta (Sin leer → Atendida → Resuelta)

---

## Arrancar los módulos de IA (opcional)

Si quieres el pipeline completo de visión (requiere GPU / cámara):

```bash
# Desde la carpeta backend/
python main.py
```

Esto levanta en paralelo:
- `Orchestrator_Process` — motor de reglas
- `Face_Process` — reconocimiento facial (ArcFace)
- `Ingestion_Process` — captura de cámara

---

## Variables de entorno (opcional)

Crea un archivo `backend/.env` para sobreescribir los valores por defecto:

```env
DATABASE_URL=postgresql://admin:admin@localhost:5432/security_db
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password123
MINIO_BUCKET=evidence
```

---

## Resumen de endpoints integrados

| Método | Ruta | Descripción |
|---|---|---|
| GET | /api/alerts/ | Listar alertas (filtro por status) |
| GET | /api/alerts/count | Conteo de alertas por estado |
| PATCH | /api/alerts/{id}/status | Cambiar estado de alerta |
| GET | /api/incidents/ | Listar incidentes |
| POST | /api/incidents/simulate | Simular evento de IA |
| GET | /api/persons/ | Listar personas enroladas |
| POST | /api/persons/ | Registrar nueva persona |
| POST | /api/persons/{id}/enroll | Enrolar biometría facial |
| GET | /api/cameras/ | Listar cámaras |
| WS | /ws/alerts | WebSocket alertas en tiempo real |
| WS | /ws | WebSocket video/detecciones |
