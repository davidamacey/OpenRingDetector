<script lang="ts">
  interface Props {
    type: string;
    size?: 'sm' | 'md';
  }
  let { type, size = 'md' }: Props = $props();

  const styles: Record<string, { bg: string; text: string }> = {
    motion:    { bg: '#1e3a5f', text: '#60a5fa' },
    arrival:   { bg: '#14532d', text: '#4ade80' },
    departure: { bg: '#450a0a', text: '#fca5a5' },
    ding:      { bg: '#4a1d96', text: '#c4b5fd' },
    unknown:   { bg: '#451a03', text: '#fcd34d' },
    vehicle:   { bg: '#134e4a', text: '#2dd4bf' },
    car:       { bg: '#134e4a', text: '#2dd4bf' },
    truck:     { bg: '#134e4a', text: '#2dd4bf' },
    person:    { bg: '#312e81', text: '#a5b4fc' },
    ok:        { bg: '#14532d', text: '#4ade80' },
    warn:      { bg: '#451a03', text: '#fcd34d' },
    fail:      { bg: '#450a0a', text: '#fca5a5' },
    missing:   { bg: '#451a03', text: '#fcd34d' },
    off:       { bg: '#1f2937', text: '#6b7280' },
    active:    { bg: '#14532d', text: '#4ade80' },
    departed:  { bg: '#1e3a5f', text: '#60a5fa' },
    face:      { bg: '#2d1b69', text: '#c4b5fd' },
  };

  const s = $derived(styles[type.toLowerCase()] ?? { bg: '#1f2937', text: '#94a3b8' });
  const px = size === 'sm' ? '0.375rem' : '0.5rem';
  const py = size === 'sm' ? '0.125rem' : '0.25rem';
  const fs = size === 'sm' ? '0.65rem' : '0.7rem';
</script>

<span
  class="inline-flex items-center font-medium rounded uppercase tracking-wide"
  style="
    background: {s.bg};
    color: {s.text};
    padding: {py} {px};
    font-size: {fs};
    letter-spacing: 0.05em;
  "
>
  {type}
</span>
