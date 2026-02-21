# Fashion Virtual Try-On Platform 👗✨

Plataforma web de prueba virtual de ropa usando IA (FASHN.ai API).

## Flujo de la aplicación

```
1. PERFIL → Usuario elige género + medidas → Se calcula talla (S/M/L)
2. CATÁLOGO → Se muestran prendas filtradas por talla y género
3. FOTO → Usuario sube foto de cuerpo completo
4. RESULTADO → IA genera foto con la ropa puesta + opción de video desfile
```

## Requisitos

- Python 3.9+
- API Key de [FASHN.ai](https://fashn.ai)

## Instalación

```bash
# 1. Instalar dependencias
pip install fastapi uvicorn python-multipart jinja2 aiofiles httpx pillow

# 2. Configurar API key (elige una opción):

# Opción A: Variable de entorno (recomendado)
export FASHN_API_KEY="tu_api_key_aqui"

# Opción B: Editar directamente en main.py línea 15
FASHN_API_KEY = "tu_api_key_aqui"

# 3. Ejecutar
cd fashion-tryon
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Abrir en navegador
# http://localhost:8000
```

## Estructura del proyecto

```
fashion-tryon/
├── main.py                 # Backend FastAPI + integración FASHN API
├── templates/
│   └── index.html          # Frontend completo (HTML + CSS + JS)
├── static/
│   ├── uploads/            # Fotos subidas (prendas y modelos)
│   └── results/            # Imágenes generadas por la IA
└── README.md
```

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Interfaz web |
| POST | `/api/upload-garment` | Subir prenda al catálogo |
| GET | `/api/catalog` | Listar prendas (filtrar por género/talla) |
| POST | `/api/estimate-size` | Calcular talla desde medidas |
| POST | `/api/try-on` | Generar prueba virtual (foto estática) |
| POST | `/api/generate-video` | Generar video de desfile |
| GET | `/api/credits` | Ver créditos FASHN restantes |

## Costos FASHN API

| Acción | Costo aproximado |
|--------|-----------------|
| Try-On (foto) | ~$0.075 USD |
| Image to Video | ~$0.15-0.30 USD |
| Model Create | ~$0.075 USD |

## Cómo subir prendas

1. Fotografiar la prenda sobre **fondo blanco**
2. Foto frontal, bien extendida, sin arrugas
3. Formato JPG o PNG
4. En la app, clic en "Agregar prenda" y llenar los datos

## Notas importantes

- Las fotos del usuario deben ser de **cuerpo completo**, de frente
- La API de FASHN soporta: tops, bottoms (pantalones/faldas), one-pieces (vestidos)
- Los resultados generados se pueden usar comercialmente
- Los datos se eliminan de FASHN después de 72 horas
