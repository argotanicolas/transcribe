# Transcribe Service

Servicio offline de transcripción de audio y video usando [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Corre completamente en tu infraestructura — ningún dato sale a servicios externos.

## Prerequisites

- Docker y Docker Compose instalados
- Instancia en Oracle Cloud (o cualquier servidor Linux)
- Puerto 8000 abierto en el Security List de Oracle Cloud

## Quick start

```bash
git clone <repo-url> transcribe
cd transcribe
cp .env.example .env
# Editar .env y cambiar API_KEY por un secreto largo y aleatorio
docker compose up -d
```

Verificar que levantó:

```bash
curl http://localhost:8000/health
```

## Endpoints

| Método | Ruta | Auth | Descripción |
|--------|------|------|-------------|
| GET | `/health` | No | Estado del servicio y modelo activo |
| POST | `/transcribe` | Bearer | Sube un archivo y devuelve transcripción completa |
| GET | `/download/{id}/txt` | Bearer | Descarga transcripción en texto plano |
| GET | `/download/{id}/srt` | Bearer | Descarga subtítulos en formato SRT |

### Auth

Todos los endpoints excepto `/health` requieren el header:

```
Authorization: Bearer YOUR_API_KEY
```

### POST /transcribe

Parámetros (multipart/form-data):

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `file` | File | Sí | Audio o video a transcribir |
| `language` | string | No | Código ISO 639-1 (ej: `es`, `en`). Si se omite, Whisper lo detecta automáticamente |

Formatos soportados: `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, `.m4a`

Respuesta:

```json
{
  "success": true,
  "job_id": "uuid",
  "language": "es",
  "duration": 123.45,
  "text": "Transcripción completa...",
  "srt_content": "1\n00:00:00,000 --> 00:00:03,500\nTexto...\n",
  "segments": [
    { "start": 0.0, "end": 3.5, "text": "Texto..." }
  ]
}
```

## Modelos disponibles

| Modelo | RAM aprox. | Calidad | Velocidad |
|--------|-----------|---------|-----------|
| `tiny` | ~1 GB | Baja | Muy rápida |
| `base` | ~1 GB | Básica | Rápida |
| `small` | ~2 GB | Buena ✅ recomendado | Media |
| `medium` | ~5 GB | Muy buena | Lenta |
| `large-v3` | ~10 GB | Excelente | Muy lenta |

Para cambiar el modelo, editá `MODEL_SIZE` en `.env` y reiniciá el contenedor:

```bash
docker compose restart
```

## Oracle Cloud — Deployment

### 1. Abrir puerto 8000

En la consola de Oracle Cloud:
- Ir a **Networking → Virtual Cloud Networks → tu VCN → Security Lists**
- Agregar Ingress Rule: Protocol TCP, Destination Port 8000, Source CIDR 0.0.0.0/0

También en la instancia:
```bash
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
```

### 2. Instalar Docker en la instancia

```bash
# Oracle Linux / Rocky Linux
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Cerrar y reabrir sesión SSH

# Docker Compose plugin
sudo dnf install -y docker-compose-plugin
```

### 3. Clonar y levantar

```bash
git clone <repo-url> ~/transcribe
cd ~/transcribe
cp .env.example .env
nano .env   # Cambiar API_KEY
docker compose up -d --build
```

### 4. Verificar

```bash
curl http://<IP_ORACLE>:8000/health
```

### 5. Actualizar

```bash
cd ~/transcribe
git pull
docker compose up -d --build
```

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `API_KEY` | — | **Requerido.** Secreto para autenticar requests |
| `MODEL_SIZE` | `small` | Tamaño del modelo Whisper |
| `COMPUTE_TYPE` | `int8` | Tipo de cómputo (`int8`, `float16`, `float32`) |
| `DEVICE` | `cpu` | `cpu` o `cuda` (si tenés GPU) |
| `MAX_FILE_SIZE_MB` | `500` | Límite de tamaño de archivo |
| `TEMP_DIR` | `/tmp/transcribe` | Directorio para archivos temporales |
