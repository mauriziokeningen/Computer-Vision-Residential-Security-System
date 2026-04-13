#!/usr/bin/env bash
# =========================================================
# start.sh — Arranca el backend FastAPI (modo desarrollo)
# Ejecutar desde la raíz del proyecto
# =========================================================
set -e

echo "▶ Iniciando backend..."
cd backend
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
