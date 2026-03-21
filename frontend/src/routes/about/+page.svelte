<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import { statusStore } from '$lib/stores/status';

  const status = $derived($statusStore);
</script>

<Header title="About" />

<div class="flex-1 overflow-y-auto p-4">
  <div class="max-w-2xl mx-auto space-y-6">
    <!-- Hero -->
    <div class="text-center py-8">
      <div
        class="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4"
        style="background: var(--color-accent);"
      >
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
          <circle cx="12" cy="13" r="3"/>
        </svg>
      </div>
      <h2 class="text-2xl font-bold mb-2" style="color: var(--color-text-primary);">OpenRingDetector</h2>
      <p class="text-sm" style="color: var(--color-text-secondary);">
        Self-hosted AI replacement for Ring's paid detection features
      </p>
      <p class="text-xs mt-1 font-mono" style="color: var(--color-text-muted);">v2.2.0</p>
    </div>

    <!-- What it does -->
    <div class="rounded-xl p-5" style="background: var(--color-card); border: 1px solid var(--color-border);">
      <h3 class="text-sm font-semibold mb-3" style="color: var(--color-text-primary);">What It Does</h3>
      <div class="space-y-2 text-sm" style="color: var(--color-text-secondary);">
        <p>
          OpenRingDetector listens for Ring doorbell motion events via Firebase push notifications,
          then runs a full AI detection pipeline on your own hardware — no cloud subscriptions needed.
        </p>
        <ul class="list-disc list-inside space-y-1 ml-2">
          <li>Detects objects (people, cars, trucks) using YOLO</li>
          <li>Identifies known vehicles via CLIP visual embeddings</li>
          <li>Recognizes known people via face detection and recognition</li>
          <li>Tracks visitor arrivals and departures automatically</li>
          <li>Sends push notifications with snapshot images via ntfy</li>
          <li>Optionally captions scenes using Gemma 3 via Ollama</li>
        </ul>
      </div>
    </div>

    <!-- Model Stack -->
    <div class="rounded-xl p-5" style="background: var(--color-card); border: 1px solid var(--color-border);">
      <h3 class="text-sm font-semibold mb-3" style="color: var(--color-text-primary);">Model Stack</h3>
      <div class="space-y-2">
        {#each [
          { name: 'YOLO11m', role: 'Object Detection', detail: 'Real-time multi-class detection (person, car, truck, etc.)' },
          { name: 'CLIP ViT-B/32', role: 'Vehicle Embedding', detail: 'OpenAI visual encoder, 512-dim vectors for vehicle matching' },
          { name: 'SCRFD-10G', role: 'Face Detection', detail: '5-point landmark detection, 95.2% WiderFace accuracy' },
          { name: 'ArcFace w600k_r50', role: 'Face Embedding', detail: '512-dim embeddings, 99.8% LFW accuracy' },
          { name: 'Gemma 3 4B', role: 'Scene Captioning', detail: 'Via Ollama, optional natural language scene descriptions' },
        ] as model}
          <div class="flex items-start gap-3 py-2 border-b last:border-0" style="border-color: var(--color-border);">
            <div class="flex-1">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium" style="color: var(--color-text-primary);">{model.name}</span>
                <span class="text-xs px-2 py-0.5 rounded" style="background: var(--color-surface); color: var(--color-text-muted);">
                  {model.role}
                </span>
              </div>
              <p class="text-xs mt-0.5" style="color: var(--color-text-muted);">{model.detail}</p>
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Tech Stack -->
    <div class="rounded-xl p-5" style="background: var(--color-card); border: 1px solid var(--color-border);">
      <h3 class="text-sm font-semibold mb-3" style="color: var(--color-text-primary);">Tech Stack</h3>
      <div class="grid grid-cols-2 gap-3 text-sm">
        {#each [
          { label: 'Backend', value: 'Python 3.12, FastAPI' },
          { label: 'Frontend', value: 'SvelteKit 2, Tailwind CSS 4' },
          { label: 'Database', value: 'PostgreSQL 17 + pgvector' },
          { label: 'ML Framework', value: 'PyTorch, ONNX Runtime' },
          { label: 'Notifications', value: 'ntfy (self-hosted)' },
          { label: 'Captioning', value: 'Ollama (Gemma 3 4B)' },
          { label: 'Ring Integration', value: 'ring-doorbell + Firebase' },
          { label: 'Containerization', value: 'Docker Compose' },
        ] as item}
          <div class="py-1.5">
            <p class="text-xs" style="color: var(--color-text-muted);">{item.label}</p>
            <p style="color: var(--color-text-secondary);">{item.value}</p>
          </div>
        {/each}
      </div>
    </div>

    <!-- How it works -->
    <div class="rounded-xl p-5" style="background: var(--color-card); border: 1px solid var(--color-border);">
      <h3 class="text-sm font-semibold mb-3" style="color: var(--color-text-primary);">How It Works</h3>
      <div class="space-y-3">
        {#each [
          { step: '1', title: 'Motion Event', desc: 'Ring doorbell detects motion and sends a Firebase push notification' },
          { step: '2', title: 'Video Capture', desc: 'Downloads snapshot immediately, then waits for the full video recording' },
          { step: '3', title: 'Frame Extraction', desc: 'Extracts key frames from the video at configurable intervals' },
          { step: '4', title: 'Object Detection', desc: 'Batched YOLO inference detects people, vehicles, and other objects' },
          { step: '5', title: 'Identity Matching', desc: 'CLIP embeddings match vehicles; SCRFD + ArcFace identifies known people' },
          { step: '6', title: 'Visit Tracking', desc: 'Records arrivals, extends visits on continued motion, detects departures' },
          { step: '7', title: 'Notification', desc: 'Sends push notification with the best snapshot frame via ntfy' },
        ] as item}
          <div class="flex gap-3">
            <div
              class="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
              style="background: var(--color-accent); color: white;"
            >
              {item.step}
            </div>
            <div>
              <p class="text-sm font-medium" style="color: var(--color-text-primary);">{item.title}</p>
              <p class="text-xs" style="color: var(--color-text-muted);">{item.desc}</p>
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Attribution -->
    <div class="rounded-xl p-5" style="background: var(--color-card); border: 1px solid var(--color-border);">
      <h3 class="text-sm font-semibold mb-3" style="color: var(--color-text-primary);">Attribution & Credits</h3>
      <div class="space-y-2 text-sm" style="color: var(--color-text-secondary);">
        <p>Built on open-source models and libraries:</p>
        <ul class="space-y-1.5 ml-2">
          <li>
            <span style="color: var(--color-text-primary);">YOLO</span> by
            <span style="color: var(--color-accent);">Ultralytics</span> — real-time object detection
          </li>
          <li>
            <span style="color: var(--color-text-primary);">CLIP</span> by
            <span style="color: var(--color-accent);">OpenAI</span> — visual-language embeddings
          </li>
          <li>
            <span style="color: var(--color-text-primary);">SCRFD</span> by
            <span style="color: var(--color-accent);">InsightFace</span> — efficient face detection
          </li>
          <li>
            <span style="color: var(--color-text-primary);">ArcFace</span> by
            <span style="color: var(--color-accent);">InsightFace</span> — face recognition embeddings
          </li>
          <li>
            <span style="color: var(--color-text-primary);">Gemma 3</span> by
            <span style="color: var(--color-accent);">Google DeepMind</span> — scene captioning
          </li>
          <li>
            <span style="color: var(--color-text-primary);">pgvector</span> — vector similarity search for PostgreSQL
          </li>
          <li>
            <span style="color: var(--color-text-primary);">ring-doorbell</span> — unofficial Ring API client
          </li>
          <li>
            <span style="color: var(--color-text-primary);">ntfy</span> by
            <span style="color: var(--color-accent);">Philipp C. Heckel</span> — push notifications
          </li>
        </ul>
      </div>
    </div>

    <!-- System info -->
    {#if status}
      <div class="rounded-xl p-5" style="background: var(--color-card); border: 1px solid var(--color-border);">
        <h3 class="text-sm font-semibold mb-3" style="color: var(--color-text-primary);">System Info</h3>
        <div class="grid grid-cols-2 gap-2 text-sm">
          <div>
            <p class="text-xs" style="color: var(--color-text-muted);">API Uptime</p>
            <p style="color: var(--color-text-secondary);">{Math.floor(status.uptime_seconds / 60)}m</p>
          </div>
          <div>
            <p class="text-xs" style="color: var(--color-text-muted);">Watcher</p>
            <p style="color: {status.watcher_running ? 'var(--color-green)' : 'var(--color-text-muted)'};">
              {status.watcher_running ? 'Running' : 'Stopped'}
            </p>
          </div>
          <div>
            <p class="text-xs" style="color: var(--color-text-muted);">Database</p>
            <p style="color: var(--color-text-secondary);">{status.database.detail}</p>
          </div>
          <div>
            <p class="text-xs" style="color: var(--color-text-muted);">GPU</p>
            <p style="color: var(--color-text-secondary);">{status.gpu.detail.split(',')[0]?.trim() || 'N/A'}</p>
          </div>
        </div>
      </div>
    {/if}

    <p class="text-center text-xs pb-6" style="color: var(--color-text-muted);">
      AGPL-3.0 License
    </p>
  </div>
</div>
