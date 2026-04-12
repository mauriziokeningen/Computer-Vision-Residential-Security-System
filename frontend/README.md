# Seguridad UH · Frontend (Vite + React + TS + Tailwind)

Este proyecto implementa el frontend de los mockups funcionales:
- Dashboard
- Monitoreo en vivo
- Alertas
- Incidentes (detalle)
- Residentes (enrolamiento)
- Control de acceso
- Configuración

## Requisitos
- Node.js 18+

## Instalación
```bash
npm install
npm run dev
```

## Build
```bash
npm run build
npm run preview
```

## Nota
Los componentes UI aquí incluidos son una **implementación ligera** inspirada en shadcn/ui para que puedas correr el proyecto
sin instalar Radix. Si quieres migrar 1:1 a shadcn/ui real, el API de componentes ya está alineado (Card, Button, etc.).


## Prueba en tiempo real (webcam + best.pt)

Este frontend se conecta al micro-backend de prueba por WebSocket en `ws://127.0.0.1:8000/ws`.

1) Levanta tu backend de prueba (weapon_server) en el puerto 8000.
2) Corre este frontend con `npm run dev`.
3) Entra a **Monitoreo en vivo** y acepta permisos de cámara.

Si cambias el puerto o ruta, edita la URL en `src/App.tsx`.
