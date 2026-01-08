# Frappe RAG - Makefile
# =====================

.PHONY: help install vllm chat serve index benchmark benchmark-smoke benchmark-stress benchmark-spike visualize clean

# vLLM Ayarları
VLLM_MODEL ?= Qwen/Qwen2.5-3B-Instruct
VLLM_PORT ?= 8282
VLLM_GPU_UTIL ?= 0.8

# Varsayılan hedef
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              Frappe RAG - Kullanılabilir Komutlar                ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  make vllm             vLLM sunucusunu başlat (:8282) [İLK ADIM] ║"
	@echo "║  make vllm-bg          vLLM'i arka planda başlat                 ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  make install          Bağımlılıkları kur                        ║"
	@echo "║  make chat             İnteraktif soru-cevap başlat              ║"
	@echo "║  make serve            RAG API sunucusunu başlat (:8088)         ║"
	@echo "║  make index            PDF'leri indeksle                         ║"
	@echo "║  make index-force      Tüm indeksi sıfırla ve yeniden oluştur    ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  make benchmark        Load testi çalıştır (5-30 kullanıcı)      ║"
	@echo "║  make benchmark-smoke  Smoke testi (1-2 kullanıcı)               ║"
	@echo "║  make benchmark-stress Stress testi (50-100 kullanıcı)           ║"
	@echo "║  make benchmark-spike  Spike testi (150-200 kullanıcı)           ║"
	@echo "║  make visualize        Benchmark sonuçlarını görselleştir        ║"
	@echo "║  make compare          Benchmark sonuçlarını karşılaştır         ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  make clean            Geçici dosyaları temizle                  ║"
	@echo "║  make clean-index      Qdrant veritabanını sil                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 Tipik Kullanım:"
	@echo "   Terminal 1: make vllm     (vLLM başlat, GPU %90)"
	@echo "   Terminal 2: make serve    (RAG API başlat)"
	@echo "   Terminal 3: make chat     (veya benchmark)"

# Bağımlılıkları kur
install:
	@echo "📦 Bağımlılıklar kuruluyor..."
	uv pip install -r requirements.txt
	@echo "✅ Kurulum tamamlandı!"

# vLLM Sunucusu (GPU %90 sınırlı)
vllm:
	@echo "🚀 vLLM sunucusu başlatılıyor..."
	@echo "   Model: $(VLLM_MODEL)"
	@echo "   Port:  $(VLLM_PORT)"
	@echo "   GPU:   $(VLLM_GPU_UTIL) (%90)"
	@echo ""
	python -m vllm.entrypoints.openai.api_server \
		--model $(VLLM_MODEL) \
		--port $(VLLM_PORT) \
		--gpu-memory-utilization $(VLLM_GPU_UTIL) \
		--trust-remote-code

vllm-bg:
	@echo "🚀 vLLM arka planda başlatılıyor..."
	nohup python -m vllm.entrypoints.openai.api_server \
		--model $(VLLM_MODEL) \
		--port $(VLLM_PORT) \
		--gpu-memory-utilization $(VLLM_GPU_UTIL) \
		--trust-remote-code > vllm.log 2>&1 &
	@echo "✅ vLLM başlatıldı. Log: vllm.log"
	@echo "   Durdurmak için: pkill -f 'vllm.entrypoints'"

# İnteraktif chat modu
chat:
	@echo "💬 Chat modu başlatılıyor..."
	python frappe_rag.py

# API sunucusu
serve:
	@echo "🚀 API sunucusu başlatılıyor (http://localhost:8088)..."
	python frappe_rag.py --serve

# İndeksleme
index:
	@echo "📄 PDF'ler indeksleniyor..."
	python frappe_rag.py --index

index-force:
	@echo "🔄 İndeks sıfırlanıyor ve yeniden oluşturuluyor..."
	python frappe_rag.py --index --force

# Benchmark komutları
benchmark:
	@echo "📊 Load testi başlatılıyor..."
	@echo "⚠️  RAG sunucusunun çalıştığından emin olun (make serve)"
	cd benchmarks && python benchmark.py --test-type load

benchmark-smoke:
	@echo "🔍 Smoke testi başlatılıyor..."
	cd benchmarks && python benchmark.py --test-type smoke

benchmark-stress:
	@echo "💪 Stress testi başlatılıyor..."
	cd benchmarks && python benchmark.py --test-type stress

benchmark-spike:
	@echo "⚡ Spike testi başlatılıyor..."
	cd benchmarks && python benchmark.py --test-type spike

# Sonuçları görselleştir
visualize:
	@echo "📈 Sonuçlar görselleştiriliyor..."
	cd benchmarks && python visualize_results.py

compare:
	@echo "🔬 Sonuçlar karşılaştırılıyor..."
	cd benchmarks && python compare_results.py

# Temizlik
clean:
	@echo "🧹 Geçici dosyalar temizleniyor..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Temizlik tamamlandı!"

clean-index:
	@echo "🗑️  Qdrant veritabanı siliniyor..."
	rm -rf qdrant_db
	@echo "✅ Veritabanı silindi. Yeniden indeksleme için: make index"
