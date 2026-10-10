package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	maxUploadBytes = 15 << 20
	maxBigBytes    = 120 << 20
	partMaxBytes   = 10 << 20
	partsMax       = 12
	maxDim         = 6000
	maxX2Input     = 3000
	maxX4Input     = 1500
)

var (
	errToolMissing = errors.New("realesrgan binary not found")
	sem            = make(chan struct{}, 2)
	workerClient   = &http.Client{Timeout: 90 * time.Second}
)

func workerBase() string {
	w := os.Getenv("ENHANCE_WORKER")
	if w == "" {
		w = "https://xdownload-api.xeraze-official.workers.dev"
	}
	return strings.TrimRight(w, "/")
}

func realSrgbDir() string {
	d := os.Getenv("REALSR_DIR")
	if d == "" {
		d = "realesrgan"
	}
	return d
}

func main() {
	addr := os.Getenv("ENHANCE_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8081"
	}
	key := os.Getenv("ENHANCE_API_KEY")

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", handleHealth)
	mux.HandleFunc("POST /api/enhance", handleEnhance)
	mux.HandleFunc("POST /api/enhance-big", handleEnhanceBig)

	var handler http.Handler = mux
	if key != "" {
		handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !hmac.Equal([]byte(r.Header.Get("X-Api-Key")), []byte(key)) {
				writeJSON(w, http.StatusForbidden, map[string]string{"error": "forbidden"})
				return
			}
			mux.ServeHTTP(w, r)
		})
	}

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("enhance server listening on http://%s (key=%v)", addr, key != "")
	if err := srv.ListenAndServe(); err != nil {
		log.Println("server:", err)
		os.Exit(1)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func handleEnhance(w http.ResponseWriter, r *http.Request) {
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	default:
		writeJSON(w, http.StatusTooManyRequests, map[string]string{"error": "busy"})
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes+(1<<20))
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "multipart expected"})
		return
	}
	fh, _, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "file required"})
		return
	}
	defer fh.Close()

	raw, err := io.ReadAll(io.LimitReader(fh, maxUploadBytes+1))
	if err != nil || len(raw) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "read failed"})
		return
	}
	if len(raw) > maxUploadBytes {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "image too big"})
		return
	}

	out, format, status, msg := enhanceBytes(raw, r.FormValue("mode"), r.FormValue("engine"))
	if status != 0 {
		writeJSON(w, status, map[string]string{"error": msg})
		return
	}
	body, ct, status, msg := encodeEnhanced(out, format)
	if status != 0 {
		writeJSON(w, status, map[string]string{"error": msg})
		return
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("X-Enhance-Engine", r.FormValue("engine"))
	_, _ = w.Write(body)
}

func enhanceBytes(raw []byte, mode, engine string) (image.Image, string, int, string) {
	if mode == "" {
		mode = "clean"
	}
	if engine == "" {
		engine = "local"
	}
	if mode != "clean" && mode != "x2" && mode != "x4" {
		return nil, "", http.StatusBadRequest, "invalid mode"
	}
	if engine != "local" && engine != "ai" {
		return nil, "", http.StatusBadRequest, "invalid engine"
	}
	if engine == "ai" && mode == "clean" {
		mode = "x2"
	}

	img, format, err := image.Decode(bytes.NewReader(raw))
	if err != nil || (format != "jpeg" && format != "png") {
		return nil, "", http.StatusUnsupportedMediaType, "jpeg or png required"
	}
	b := img.Bounds()
	inW, inH := b.Dx(), b.Dy()
	if inW > maxDim || inH > maxDim || inW < 16 || inH < 16 {
		return nil, "", http.StatusBadRequest, "unsupported dimensions"
	}
	if mode == "x2" && (inW > maxX2Input || inH > maxX2Input) {
		return nil, "", http.StatusBadRequest, "x2 input too large"
	}
	if mode == "x4" && (inW > maxX4Input || inH > maxX4Input) {
		return nil, "", http.StatusBadRequest, "x4 input too large"
	}
	targetW, targetH := inW, inH
	if mode == "x2" {
		targetW, targetH = inW*2, inH*2
	}
	if mode == "x4" {
		targetW, targetH = inW*4, inH*4
	}

	var out image.Image
	if engine == "ai" {
		data, err := realESRGANEnhance(raw)
		if err != nil {
			if errors.Is(err, errToolMissing) {
				return nil, "", http.StatusServiceUnavailable, "ai unavailable"
			}
			return nil, "", http.StatusBadGateway, "ai failed"
		}
		res, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, "", http.StatusBadGateway, "ai decode failed"
		}
		out = lanczosResize(res, targetW, targetH)
	} else {
		out = localEnhance(img, mode, targetW, targetH)
	}
	return out, format, 0, ""
}

func encodeEnhanced(out image.Image, format string) ([]byte, string, int, string) {
	buf := &bytes.Buffer{}
	ct := "image/jpeg"
	if format == "png" {
		ct = "image/png"
		enc := png.Encoder{CompressionLevel: png.BestSpeed}
		if err := enc.Encode(buf, out); err != nil {
			return nil, "", http.StatusInternalServerError, "encode failed"
		}
	} else {
		if err := jpeg.Encode(buf, out, &jpeg.Options{Quality: 95}); err != nil {
			return nil, "", http.StatusInternalServerError, "encode failed"
		}
	}
	return buf.Bytes(), ct, 0, ""
}

func validID(s string) bool {
	if len(s) < 8 || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if c != '-' && (c < '0' || c > '9') && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') {
			return false
		}
	}
	return true
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func fetchPart(fileID string, i int) ([]byte, error) {
	url := workerBase() + "/api/enhance-part/" + fileID + "/" + strconv.Itoa(i)
	var lastErr error
	for attempt := 0; attempt < 8; attempt++ {
		if attempt > 0 {
			time.Sleep(1500 * time.Millisecond)
		}
		resp, err := workerClient.Get(url)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode == http.StatusOK {
			data, err := io.ReadAll(io.LimitReader(resp.Body, partMaxBytes+1))
			resp.Body.Close()
			if err != nil {
				lastErr = err
				continue
			}
			return data, nil
		}
		resp.Body.Close()
		if resp.StatusCode == http.StatusNotFound {
			lastErr = fmt.Errorf("part %d not ready", i)
			continue
		}
		return nil, fmt.Errorf("part %d: status %d", i, resp.StatusCode)
	}
	return nil, lastErr
}

func putPart(fileID string, i int, data []byte) error {
	url := workerBase() + "/api/enhance-part/" + fileID + "/" + strconv.Itoa(i)
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Second)
		}
		req, err := http.NewRequest(http.MethodPut, url, bytes.NewReader(data))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/octet-stream")
		resp, err := workerClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			return nil
		}
		lastErr = fmt.Errorf("put part %d: status %d", i, resp.StatusCode)
	}
	return lastErr
}

func handleEnhanceBig(w http.ResponseWriter, r *http.Request) {
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	default:
		writeJSON(w, http.StatusTooManyRequests, map[string]string{"error": "busy"})
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	var req struct {
		FileID string `json:"fileId"`
		Parts  int    `json:"parts"`
		Mode   string `json:"mode"`
		Engine string `json:"engine"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	if !validID(req.FileID) || req.Parts < 1 || req.Parts > partsMax {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid upload"})
		return
	}

	raw := make([]byte, 0, req.Parts*partMaxBytes)
	for i := 0; i < req.Parts; i++ {
		part, err := fetchPart(req.FileID, i)
		if err != nil {
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": "assemble failed"})
			return
		}
		raw = append(raw, part...)
		if len(raw) > maxBigBytes {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "image too big"})
			return
		}
	}

	out, format, status, msg := enhanceBytes(raw, req.Mode, req.Engine)
	if status != 0 {
		writeJSON(w, status, map[string]string{"error": msg})
		return
	}
	body, ct, status, msg := encodeEnhanced(out, format)
	if status != 0 {
		writeJSON(w, status, map[string]string{"error": msg})
		return
	}
	resultParts := (len(body) + partMaxBytes - 1) / partMaxBytes
	if resultParts > partsMax {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "result too big"})
		return
	}
	resultID := randomHex(16)
	for i := 0; i < resultParts; i++ {
		start := i * partMaxBytes
		end := min(start+partMaxBytes, len(body))
		if err := putPart(resultID, i, body[start:end]); err != nil {
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": "result store failed"})
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"resultId": resultID, "parts": resultParts, "type": ct})
}

func realESRGANEnhance(src []byte) ([]byte, error) {
	dir := realSrgbDir()
	exe := filepath.Join(dir, "realesrgan-ncnn-vulkan.exe")
	if _, err := os.Stat(exe); err != nil {
		return nil, errToolMissing
	}
	tmp, err := os.MkdirTemp("", "enh-esr-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	inPath := filepath.Join(tmp, "in.png")
	outPath := filepath.Join(tmp, "out.png")
	if err := os.WriteFile(inPath, src, 0o644); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-i", inPath, "-o", outPath, "-n", "realesrgan-x4plus", "-s", "4", "-f", "png")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("realesrgan: %w: %s", err, string(output))
	}
	return os.ReadFile(outPath)
}

func localEnhance(src image.Image, mode string, targetW, targetH int) image.Image {
	rgba := toRGBA(src)
	if mode == "clean" {
		median3(rgba)
		unsharp(rgba, 0.55, 1)
		if rgba.Bounds().Dx() != targetW || rgba.Bounds().Dy() != targetH {
			return resizeTo(rgba, targetW, targetH)
		}
		return rgba
	}
	up := lanczos2x(rgba)
	if mode == "x4" {
		up = lanczos2x(up)
	}
	unsharp(up, 0.4, 1)
	if up.Bounds().Dx() != targetW || up.Bounds().Dy() != targetH {
		return resizeTo(up, targetW, targetH)
	}
	return up
}

func toRGBA(src image.Image) *image.RGBA {
	if r, ok := src.(*image.RGBA); ok {
		return r
	}
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			dst.Set(x, y, src.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return dst
}

func median3(img *image.RGBA) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	src := make([]byte, len(img.Pix))
	copy(src, img.Pix)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			var r, g, bb, a [9]byte
			i := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					o := (y+dy)*img.Stride + (x+dx)*4
					r[i], g[i], bb[i], a[i] = src[o], src[o+1], src[o+2], src[o+3]
					i++
				}
			}
			o := y*img.Stride + x*4
			img.Pix[o] = med9(r)
			img.Pix[o+1] = med9(g)
			img.Pix[o+2] = med9(bb)
			img.Pix[o+3] = med9(a)
		}
	}
}

func med9(v [9]byte) byte {
	for i := 1; i < 9; i++ {
		key := v[i]
		j := i - 1
		for j >= 0 && v[j] > key {
			v[j+1] = v[j]
			j--
		}
		v[j+1] = key
	}
	return v[4]
}

func boxBlur(img *image.RGBA, radius int) *image.RGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	tmp := image.NewRGBA(image.Rect(0, 0, w, h))
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var sr, sg, sb, sa, n int
			for k := -radius; k <= radius; k++ {
				xx := x + k
				if xx < 0 {
					xx = 0
				} else if xx >= w {
					xx = w - 1
				}
				o := y*img.Stride + xx*4
				sr += int(img.Pix[o])
				sg += int(img.Pix[o+1])
				sb += int(img.Pix[o+2])
				sa += int(img.Pix[o+3])
				n++
			}
			o := y*tmp.Stride + x*4
			tmp.Pix[o] = byte(sr / n)
			tmp.Pix[o+1] = byte(sg / n)
			tmp.Pix[o+2] = byte(sb / n)
			tmp.Pix[o+3] = byte(sa / n)
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var sr, sg, sb, sa, n int
			for k := -radius; k <= radius; k++ {
				yy := y + k
				if yy < 0 {
					yy = 0
				} else if yy >= h {
					yy = h - 1
				}
				o := yy*tmp.Stride + x*4
				sr += int(tmp.Pix[o])
				sg += int(tmp.Pix[o+1])
				sb += int(tmp.Pix[o+2])
				sa += int(tmp.Pix[o+3])
				n++
			}
			o := y*out.Stride + x*4
			out.Pix[o] = byte(sr / n)
			out.Pix[o+1] = byte(sg / n)
			out.Pix[o+2] = byte(sb / n)
			out.Pix[o+3] = byte(sa / n)
		}
	}
	return out
}

func unsharp(img *image.RGBA, amount float64, radius int) {
	blur := boxBlur(img, radius)
	n := len(img.Pix)
	for i := 0; i < n; i += 4 {
		for c := 0; c < 3; c++ {
			orig := float64(img.Pix[i+c])
			d := orig + amount*(orig-float64(blur.Pix[i+c]))
			if d < 0 {
				d = 0
			} else if d > 255 {
				d = 255
			}
			img.Pix[i+c] = byte(d)
		}
	}
}

func lanczosKernel(t float64) float64 {
	if t == 0 {
		return 1
	}
	if t <= -3 || t >= 3 {
		return 0
	}
	pt := math.Pi * t
	return (3 * math.Sin(pt) * math.Sin(pt/3)) / (pt * pt)
}

func lanczosResize(src image.Image, tw, th int) image.Image {
	rgba := toRGBA(src)
	if tw <= 0 || th <= 0 {
		return rgba
	}
	b := rgba.Bounds()
	if b.Dx() == tw && b.Dy() == th {
		return rgba
	}
	mid := lanczosResample(rgba, tw, b.Dy(), false)
	return lanczosResample(mid, tw, th, true)
}

func lanczos2x(src *image.RGBA) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dw, dh := w*2, h*2
	horiz := lanczosResample(src, dw, h, false)
	return lanczosResample(horiz, dw, dh, true)
}

func lanczosResample(src *image.RGBA, dw, dh int, vertical bool) *image.RGBA {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	if !vertical {
		scale := float64(sw) / float64(dw)
		for y := 0; y < sh; y++ {
			for x := 0; x < dw; x++ {
				center := (float64(x)+0.5)*scale - 0.5
				lo := int(math.Ceil(center - 3))
				hi := int(math.Floor(center + 3))
				var wr, wg, wb, wa float64
				for sx := lo; sx <= hi; sx++ {
					c := sx
					if c < 0 {
						c = 0
					} else if c >= sw {
						c = sw - 1
					}
					ww := lanczosKernel((float64(sx) - center) / scale)
					if ww == 0 {
						continue
					}
					o := y*src.Stride + c*4
					a := float64(src.Pix[o+3]) / 255
					wr += ww * float64(src.Pix[o]) * a
					wg += ww * float64(src.Pix[o+1]) * a
					wb += ww * float64(src.Pix[o+2]) * a
					wa += ww * a
				}
				o := y*dst.Stride + x*4
				dst.Pix[o+3] = clamp255(wa * 255)
				if wa > 0 {
					dst.Pix[o] = clamp255(wr / wa)
					dst.Pix[o+1] = clamp255(wg / wa)
					dst.Pix[o+2] = clamp255(wb / wa)
				}
			}
		}
		return dst
	}
	scale := float64(sh) / float64(dh)
	for y := 0; y < dh; y++ {
		center := (float64(y)+0.5)*scale - 0.5
		lo := int(math.Ceil(center - 3))
		hi := int(math.Floor(center + 3))
		for x := 0; x < dw; x++ {
			var wr, wg, wb, wa float64
			for sy := lo; sy <= hi; sy++ {
				c := sy
				if c < 0 {
					c = 0
				} else if c >= sh {
					c = sh - 1
				}
				ww := lanczosKernel((float64(sy) - center) / scale)
				if ww == 0 {
					continue
				}
				o := c*src.Stride + x*4
				a := float64(src.Pix[o+3]) / 255
				wr += ww * float64(src.Pix[o]) * a
				wg += ww * float64(src.Pix[o+1]) * a
				wb += ww * float64(src.Pix[o+2]) * a
				wa += ww * a
			}
			o := y*dst.Stride + x*4
			dst.Pix[o+3] = clamp255(wa * 255)
			if wa > 0 {
				dst.Pix[o] = clamp255(wr / wa)
				dst.Pix[o+1] = clamp255(wg / wa)
				dst.Pix[o+2] = clamp255(wb / wa)
			}
		}
	}
	return dst
}

func clamp255(v float64) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v + 0.5)
}

func resizeTo(img image.Image, tw, th int) image.Image {
	b := img.Bounds()
	if b.Dx() == tw && b.Dy() == th {
		return img
	}
	src := toRGBA(img)
	sw, sh := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, tw, th))
	xs := float64(sw) / float64(tw)
	ys := float64(sh) / float64(th)
	for y := 0; y < th; y++ {
		sy := int((float64(y)+0.5)*ys - 0.5)
		if sy < 0 {
			sy = 0
		} else if sy >= sh {
			sy = sh - 1
		}
		for x := 0; x < tw; x++ {
			sx := int((float64(x)+0.5)*xs - 0.5)
			if sx < 0 {
				sx = 0
			} else if sx >= sw {
				sx = sw - 1
			}
			o := sy*src.Stride + sx*4
			d := y*dst.Stride + x*4
			dst.Pix[d] = src.Pix[o]
			dst.Pix[d+1] = src.Pix[o+1]
			dst.Pix[d+2] = src.Pix[o+2]
			dst.Pix[d+3] = src.Pix[o+3]
		}
	}
	return dst
}
