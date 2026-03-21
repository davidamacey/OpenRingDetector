<script lang="ts">
  import { page } from '$app/stores';
  import {
    Activity,
    BarChart3,
    Camera,
    Car,
    ChevronLeft,
    ChevronRight,
    Clock,
    HelpCircle,
    Settings,
    Users
  } from 'lucide-svelte';

  let collapsed = $state(false);

  const navItems = [
    { href: '/', label: 'Live Feed', icon: Activity },
    { href: '/history', label: 'History', icon: Clock },
    { href: '/vehicles', label: 'Vehicles', icon: Car },
    { href: '/faces', label: 'Faces', icon: Users },
    { href: '/unmatched', label: 'Unmatched', icon: HelpCircle },
    { href: '/analytics', label: 'Analytics', icon: BarChart3 },
    { href: '/settings', label: 'Settings', icon: Settings }
  ];
</script>

<aside
  class="flex flex-col h-full transition-all duration-200 border-r"
  style="
    width: {collapsed ? '56px' : '220px'};
    background: var(--color-surface);
    border-color: var(--color-border);
    min-width: {collapsed ? '56px' : '220px'};
  "
>
  <!-- Logo -->
  <a
    href="/about"
    class="flex items-center gap-3 px-4 py-4 border-b transition-colors"
    style="border-color: var(--color-border); min-height: 60px; text-decoration: none;"
    title="About OpenRingDetector"
  >
    <div
      class="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
      style="background: var(--color-accent);"
    >
      <Camera size={16} color="white" />
    </div>
    {#if !collapsed}
      <span class="font-semibold text-sm leading-tight" style="color: var(--color-text-primary);">
        OpenRing<br />Detector
      </span>
    {/if}
  </a>

  <!-- Nav -->
  <nav class="flex-1 py-3 overflow-y-auto">
    {#each navItems as item}
      {@const active = $page.url.pathname === item.href || ($page.url.pathname.startsWith(item.href) && item.href !== '/')}
      <a
        href={item.href}
        class="flex items-center gap-3 px-4 py-2.5 mx-2 rounded-lg text-sm font-medium transition-colors mb-0.5"
        style="
          background: {active ? 'rgba(59,130,246,0.15)' : 'transparent'};
          color: {active ? 'var(--color-accent)' : 'var(--color-text-secondary)'};
        "
        title={collapsed ? item.label : ''}
      >
        <svelte:component this={item.icon} size={18} />
        {#if !collapsed}
          <span>{item.label}</span>
        {/if}
      </a>
    {/each}
  </nav>

  <!-- Collapse toggle -->
  <button
    onclick={() => (collapsed = !collapsed)}
    class="flex items-center justify-center py-3 border-t w-full transition-colors"
    style="
      border-color: var(--color-border);
      color: var(--color-text-muted);
      background: transparent;
    "
  >
    {#if collapsed}
      <ChevronRight size={16} />
    {:else}
      <ChevronLeft size={16} />
    {/if}
  </button>
</aside>
