# Molfun Benchmarks Dashboard

Dashboard profesional para ejecutar y visualizar benchmarks de kernels Triton.

## Estructura del Proyecto

```
molfun/
├── src/
│   ├── api/              # API FastAPI
│   ├── benchmarks/       # Benchmarks disponibles
│   └── kernels/          # Kernels Triton
├── frontend/             # Dashboard Next.js
└── data/                 # Resultados guardados
```

## Instalación y Ejecución

### Backend (FastAPI)

1. Instalar dependencias:
```bash
pip install -r src/api/requirements.txt
```

2. Ejecutar la API:
```bash
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

O usar el script:
```bash
chmod +x src/api/run_api.sh
./src/api/run_api.sh
```

La API estará disponible en `http://localhost:8000`

### Frontend (Next.js)

1. Instalar dependencias:
```bash
cd frontend
npm install
```

2. Ejecutar el servidor de desarrollo:
```bash
npm run dev
```

El dashboard estará disponible en `http://localhost:3000`

## Uso

1. Inicia el backend (FastAPI) en el puerto 8000
2. Inicia el frontend (Next.js) en el puerto 3000
3. Abre el navegador en `http://localhost:3000`
4. Selecciona un benchmark y haz clic en "Run"
5. Espera a que se complete (el frontend mostrará un loader)
6. Visualiza los resultados con gráficos y tablas

## Endpoints de la API

- `GET /benchmarks` - Lista todos los benchmarks disponibles
- `POST /benchmarks/{benchmark_name}/run` - Ejecuta un benchmark específico
- `GET /results` - Lista todos los resultados guardados
- `GET /results/{filename}` - Obtiene un resultado específico

## Características

- ✅ Detección automática de benchmarks disponibles
- ✅ Ejecución de benchmarks con loader en tiempo real
- ✅ Visualización de resultados con gráficos (speedup, comparación de tiempos)
- ✅ Tabla detallada con métricas de correctness
- ✅ Guardado automático de resultados en `data/`
- ✅ Dashboard profesional con Tailwind CSS y Recharts
