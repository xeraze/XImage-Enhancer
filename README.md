# XImageEnhancer

Self-hosted image enhancement HTTP server.

Three engines:

- **local** - denoise + unsharp + Lanczos upscale (1:1 / ×2 / ×4). Always available, CPU-only.
- **ai** - shells out to [Real-ESRGAN ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) (a prebuilt C++ binary) for ×4 neural upscale, then resizes to the requested target. Requires the binary to be present.
- **redraw** (optional, separate Python service) - true generative redraw using [SUPIR](https://github.com/Kijai/SUPIR) (SDXL-based diffusion). See [`redraw/`](redraw/) for setup. Requires an NVIDIA GPU with ≥ 10 GB VRAM.

Originally extracted from the [XDownload](https://github.com/xeraze/XDownload-Web) web project's enhancement feature.

## Requirements

- Go 1.22+ (to build)
- (optional, for `engine=ai`) `realesrgan-ncnn-vulkan` binary + models - [download](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases)

## Quick start

```bash
git clone https://github.com/xeraze/go-image-enhance.git
cd go-image-enhance
go build -o enhance-server .

 local-only (no extra binaries needed)
./enhance-server

 with AI engine
REALSR_DIR=/path/to/realesrgan ./enhance-server
```

Server listens on `127.0.0.1:8081` by default. Override with `ENHANCE_ADDR`.

## API

All endpoints are JSON in / image out (except health).

### `GET /api/health`

```json
{ "ok": true }
```

### `POST /api/enhance`

Multipart form:

| field    | values                 | default  |
|----------|------------------------|----------|
| `file`   | JPEG or PNG (≤ 15 MB)  | required |
| `mode`   | `clean` \| `x2` \| `x4`| `clean`  |
| `engine` | `local` \| `ai`        | `local`  |

Constraints:

- `clean`: any size up to 6000 px per side
- `x2`: input long side ≤ 3000 px
- `x4`: input long side ≤ 1500 px

Returns the enhanced image (same format as input). Extra header `X-Enhance-Engine` echoes the engine used.

```bash
curl -F "file=@photo.jpg" -F "mode=x2" -F "engine=local" \
  http://127.0.0.1:8081/api/enhance -o photo_x2.jpg
```

### `POST /api/enhance-big`

For images larger than 15 MB (up to 120 MB). The client splits the file into ≤ 12 parts of ≤ 10 MB each, uploads them via a Cloudflare Worker KV relay, then calls this endpoint with a JSON body:

```json
{ "fileId": "<uuid>", "parts": 3, "mode": "x2", "engine": "local" }
```

The server reassembles the parts, enhances, stores the result back through the relay, and returns:

```json
{ "resultId": "<hex>", "parts": 2, "type": "image/jpeg" }
```

Set `ENHANCE_WORKER` to point at your relay (defaults to the original XDownload worker). The relay itself is a separate Cloudflare Worker - see the worker code in the original project if you need the full big-upload path.

## Environment variables

| variable           | default                                              | purpose                          |
|--------------------|------------------------------------------------------|----------------------------------|
| `ENHANCE_ADDR`     | `127.0.0.1:8081`                                     | listen address                   |
| `ENHANCE_API_KEY`  | *(empty - auth disabled)*                            | require `X-Api-Key` header       |
| `ENHANCE_WORKER`   | `https://xdownload-api.xeraze-official.workers.dev`  | relay base for big uploads       |
| `REALSR_DIR`       | `realesrgan`                                         | directory with Real-ESRGAN binary|

## Rate limiting & concurrency

- Max 2 concurrent enhance jobs (returns `429` when full)
- Multipart parse capped at 15 MB + 1 MB overhead
- Big-assemble capped at 120 MB total
- Optional HMAC API-key gate in front of all routes

## Notes

- `engine=ai` currently always uses the `realesrgan-x4plus` model at ×4, then Lanczos-resizes down to the requested target if `mode=x2`. If the binary is missing, the endpoint returns `503`.
- `clean` mode keeps dimensions and applies median denoise + unsharp mask.
- No database, no temp files left behind (everything runs in `os.MkdirTemp` and is cleaned up).