import os
import json
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Molfun Benchmarks API")

# CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
KERNELS_DIR = Path(__file__).parent.parent / "kernels"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Mapeo de benchmarks a kernels
BENCHMARK_TO_KERNEL = {
    "gelu": "gelu_triton.py",
    "fused_linear_gelu": "fused_linear_gelu_triton.py",
    "esm_fused_mlp1_synth": "fused_linear_gelu_triton.py",
    "esm_fused_mlp1_synth_t30": "fused_linear_gelu_triton.py",
    "esmfold_synth": "fused_linear_gelu_triton.py",
}


class BenchmarkResult(BaseModel):
    benchmark_name: str
    case_name: str
    baseline_time_ms: float
    triton_time_ms: float
    speedup: float
    max_diff: float
    mean_diff: float
    metadata: Dict[str, Any]


class BenchmarkRun(BaseModel):
    benchmark_name: str
    timestamp: str
    results: List[BenchmarkResult]
    total_cases: int
    success: bool
    error: Optional[str] = None


def discover_benchmarks() -> List[str]:
    """Descubre todos los benchmarks disponibles en molfun/benchmarks/"""
    benchmarks = []
    for file in BENCHMARKS_DIR.glob("bench_*.py"):
        benchmark_name = file.stem.replace("bench_", "")
        benchmarks.append(benchmark_name)
    return sorted(benchmarks)


def run_benchmark(benchmark_name: str) -> BenchmarkRun:
    """Ejecuta un benchmark y devuelve los resultados estructurados"""
    benchmark_file = BENCHMARKS_DIR / f"bench_{benchmark_name}.py"
    
    if not benchmark_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Benchmark '{benchmark_name}' no encontrado"
        )
    
    try:
        # Importar el módulo del benchmark dinámicamente
        module_name = f"bench_{benchmark_name}"
        spec = importlib.util.spec_from_file_location(
            module_name, 
            benchmark_file
        )
        module = importlib.util.module_from_spec(spec)
        
        # Registrar el módulo en sys.modules antes de ejecutarlo
        # Esto es necesario para que decoradores como @dataclass funcionen correctamente
        sys.modules[module_name] = module
        
        # Ejecutar el benchmark (debe tener función run_benchmark que devuelva resultados)
        spec.loader.exec_module(module)
        
        # Intentar usar run_benchmark_api() si existe (para compatibilidad con benchmarks que requieren argumentos)
        # Si no existe, usar run_benchmark() sin argumentos
        if hasattr(module, "run_benchmark_api"):
            results = module.run_benchmark_api()
        elif hasattr(module, "run_benchmark"):
            # Verificar si run_benchmark requiere argumentos inspeccionando su firma
            import inspect
            sig = inspect.signature(module.run_benchmark)
            if len(sig.parameters) == 0:
                results = module.run_benchmark()
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Benchmark '{benchmark_name}' requiere argumentos. Implementa run_benchmark_api() sin argumentos."
                )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Benchmark '{benchmark_name}' no tiene función run_benchmark() o run_benchmark_api()"
            )
        
        # Crear objeto BenchmarkRun
        benchmark_run = BenchmarkRun(
            benchmark_name=benchmark_name,
            timestamp=datetime.now().isoformat(),
            results=results,
            total_cases=len(results),
            success=True
        )
        
        # Guardar resultados en data/
        save_results(benchmark_run)
        
        return benchmark_run
        
    except Exception as e:
        error_msg = f"Error ejecutando benchmark: {str(e)}\n{traceback.format_exc()}"
        return BenchmarkRun(
            benchmark_name=benchmark_name,
            timestamp=datetime.now().isoformat(),
            results=[],
            total_cases=0,
            success=False,
            error=error_msg
        )


def save_results(benchmark_run: BenchmarkRun):
    """Guarda los resultados en data/"""
    filename = f"{benchmark_run.benchmark_name}_{benchmark_run.timestamp.replace(':', '-')}.json"
    filepath = DATA_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(benchmark_run.dict(), f, indent=2)


@app.get("/")
async def root():
    return {"message": "Molfun Benchmarks API"}


@app.get("/benchmarks", response_model=List[str])
async def list_benchmarks():
    """Lista todos los benchmarks disponibles"""
    return discover_benchmarks()


@app.post("/benchmarks/{benchmark_name}/run", response_model=BenchmarkRun)
async def run_benchmark_endpoint(benchmark_name: str):
    """Ejecuta un benchmark específico"""
    return run_benchmark(benchmark_name)


@app.get("/results")
async def list_results():
    """Lista todos los resultados guardados"""
    results = []
    for file in sorted(DATA_DIR.glob("*.json"), reverse=True):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                results.append({
                    "filename": file.name,
                    "benchmark_name": data.get("benchmark_name"),
                    "timestamp": data.get("timestamp"),
                    "total_cases": data.get("total_cases"),
                    "success": data.get("success")
                })
        except Exception as e:
            continue
    return results


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Obtiene un resultado específico por nombre de archivo"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Resultado no encontrado")
    
    with open(filepath, "r") as f:
        return json.load(f)


@app.get("/kernels/{benchmark_name}")
async def get_kernel_code(benchmark_name: str):
    """Obtiene el código del benchmark y del kernel asociado"""
    result = {
        "benchmark_name": benchmark_name,
        "benchmark_code": None,
        "benchmark_filename": None,
        "kernel_code": None,
        "kernel_filename": None,
    }
    
    # Obtener código del benchmark
    benchmark_file = BENCHMARKS_DIR / f"bench_{benchmark_name}.py"
    if benchmark_file.exists():
        with open(benchmark_file, "r", encoding="utf-8") as f:
            result["benchmark_code"] = f.read()
            result["benchmark_filename"] = f"bench_{benchmark_name}.py"
    
    # Obtener código del kernel asociado
    if benchmark_name in BENCHMARK_TO_KERNEL:
        kernel_filename = BENCHMARK_TO_KERNEL[benchmark_name]
        kernel_path = KERNELS_DIR / kernel_filename
        
        if kernel_path.exists():
            with open(kernel_path, "r", encoding="utf-8") as f:
                result["kernel_code"] = f.read()
                result["kernel_filename"] = kernel_filename
    
    # Si no hay ningún código disponible, devolver error
    if not result["benchmark_code"] and not result["kernel_code"]:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró benchmark ni kernel para '{benchmark_name}'"
        )
    
    return result
