<script lang="ts">
  import { User, Car } from 'lucide-svelte';
  import type { FaceProfileResponse, ReferenceResponse } from '$lib/types';

  type Profile = ReferenceResponse | FaceProfileResponse;

  interface Props {
    profile: Profile;
    type: 'vehicle' | 'face';
    onedit?: (p: Profile) => void;
  }
  let { profile, type, onedit }: Props = $props();

  const lastSeenStr = $derived(profile.last_seen
    ? new Date(profile.last_seen).toLocaleDateString()
    : 'Never');
</script>

<div
  class="rounded-xl overflow-hidden transition-colors group"
  style="background: var(--color-card); border: 1px solid var(--color-border);"
>
  <!-- Image -->
  <div
    class="h-36 flex items-center justify-center"
    style="background: var(--color-surface);"
  >
    {#if profile.sample_image_url}
      <img src={profile.sample_image_url} alt={profile.display_name} class="h-full w-full object-cover" />
    {:else}
      <div style="color: var(--color-text-muted);">
        {#if type === 'face'}
          <User size={40} />
        {:else}
          <Car size={40} />
        {/if}
      </div>
    {/if}
  </div>

  <!-- Info -->
  <div class="p-3">
    <p class="text-sm font-semibold mb-0.5 truncate" style="color: var(--color-text-primary);">
      {profile.display_name}
    </p>
    <p class="text-xs" style="color: var(--color-text-secondary);">
      {profile.visit_count} visits · {lastSeenStr}
    </p>
    <button
      onclick={() => onedit?.(profile)}
      class="mt-2 text-xs px-3 py-1 rounded w-full transition-colors"
      style="
        background: var(--color-border);
        color: var(--color-text-secondary);
      "
    >
      Edit
    </button>
  </div>
</div>
