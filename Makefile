.PHONY: help api kill-api install-api run-api kill-frontend frontend install-frontend dev test test-cov install-test

# Variables
API_PORT := 8000
FRONTEND_PORT := 3000
API_DIR := molfun/api
FRONTEND_DIR := frontend

help: ## Muestra esta ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

kill-api: ## Mata cualquier proceso en el puerto 8000
	@echo "Buscando procesos en el puerto $(API_PORT)..."
	@lsof -ti:$(API_PORT) | xargs -r kill -9 2>/dev/null || echo "No hay procesos en el puerto $(API_PORT)"
	@echo "Puerto $(API_PORT) liberado"

install-api: ## Instala dependencias de la API
	@echo "Instalando dependencias de la API..."
	pip install -r $(API_DIR)/requirements.txt

run-api: kill-api ## Levanta la API FastAPI (mata procesos existentes primero)
	@echo "Iniciando API en puerto $(API_PORT)..."
	cd $(API_DIR) && python -m uvicorn main:app --reload --host 0.0.0.0 --port $(API_PORT)

api: run-api ## Alias para run-api

install-frontend: ## Instala dependencias del frontend
	@echo "Instalando dependencias del frontend..."
	cd $(FRONTEND_DIR) && npm install

kill-frontend: ## Mata cualquier proceso en el puerto 3000
	@echo "Buscando procesos en el puerto $(FRONTEND_PORT)..."
	@lsof -ti:$(FRONTEND_PORT) | xargs -r kill -9 2>/dev/null || echo "No hay procesos en el puerto $(FRONTEND_PORT)"
	@echo "Puerto $(FRONTEND_PORT) liberado"

frontend: kill-frontend ## Levanta el frontend Next.js (mata procesos existentes primero)
	@echo "Iniciando frontend en puerto $(FRONTEND_PORT)..."
	cd $(FRONTEND_DIR) && npm run dev

dev: api frontend ## Levanta tanto la API como el frontend (en paralelo)

test: ## Ejecuta los tests
	@echo "Ejecutando tests..."
	pytest tests/ -v

test-cov: ## Ejecuta los tests con coverage
	@echo "Ejecutando tests con coverage..."
	pytest tests/ --cov=molfun --cov-report=html --cov-report=term

install-test: ## Instala dependencias de testing
	@echo "Instalando dependencias de testing..."
	pip install -r requirements-test.txt
