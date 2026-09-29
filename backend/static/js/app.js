/**
 * TideTrace Master Application Coordinator
 * Connects REST API endpoints (/api/scenes, /api/run, /api/report)
 * to MapEngine, TimelineEngine, and UIEngine state.
 */

const AppEngine = (function () {
  let activeJobId = null;
  let activeSceneId = 'scene_mumbai_001';
  let activePipelineResult = null;
  let scenesCatalog = [];

  function init() {
    console.log('[TideTrace] Initializing Naval Tactical Console...');

    // 1. Initialize Subsystems
    if (window.MapEngine) MapEngine.initMap();
    if (window.TimelineEngine) TimelineEngine.initTimeline();
    if (window.UIEngine) UIEngine.initUI();

    // 2. Fetch Health Status & Air-Gapped Verification
    fetchHealthStatus();

    // 3. Load Scene Catalog & Auto-Run Default Scene
    loadScenesAndInit();

    // 4. Bind Global Action Buttons
    bindEventHandlers();
    bindSarPreviewControls();
    bindWorkspaceControls();
  }

  let isLiveMode = false;
  const workspaceState = { leftOpen: true, rightOpen: true, focus: false, fullScreen: false, previous: null };

  function refreshMapLayout() {
    [0, 160, 320].forEach((delay) => {
      window.setTimeout(() => {
        if (window.MapEngine && MapEngine.resizeMap) MapEngine.resizeMap();
      }, delay);
    });
  }

  function applyWorkspaceState() {
    const root = document.getElementById('app-container');
    const leftToggle = document.getElementById('toggle-left-panel');
    const rightToggle = document.getElementById('toggle-right-panel');
    const focusToggle = document.getElementById('btn-focus-map');
    const fullScreenToggle = document.getElementById('btn-map-fullscreen');
    if (!root) return;

    root.classList.toggle('left-panel-collapsed', !workspaceState.leftOpen);
    root.classList.toggle('right-panel-collapsed', !workspaceState.rightOpen);
    root.classList.toggle('focus-map', workspaceState.focus);
    root.classList.toggle('map-only', workspaceState.fullScreen);

    if (leftToggle) {
      leftToggle.textContent = workspaceState.leftOpen ? '‹' : '›';
      leftToggle.setAttribute('aria-expanded', String(workspaceState.leftOpen));
      leftToggle.title = workspaceState.leftOpen ? 'Collapse suspect leaderboard' : 'Show suspect leaderboard';
    }
    if (rightToggle) {
      rightToggle.textContent = workspaceState.rightOpen ? '›' : '‹';
      rightToggle.setAttribute('aria-expanded', String(workspaceState.rightOpen));
      rightToggle.title = workspaceState.rightOpen ? 'Collapse telemetry and trace inspector' : 'Show telemetry and trace inspector';
    }
    if (focusToggle) {
      focusToggle.textContent = workspaceState.focus ? 'Exit focus' : 'Focus map';
      focusToggle.setAttribute('aria-pressed', String(workspaceState.focus));
      focusToggle.title = workspaceState.focus ? 'Restore map workspace' : 'Focus map: hide both side panels';
    }
    if (fullScreenToggle) {
      fullScreenToggle.setAttribute('aria-pressed', String(workspaceState.fullScreen));
      fullScreenToggle.title = workspaceState.fullScreen ? 'Restore full analysis workspace' : 'Show map only';
      fullScreenToggle.querySelector('.map-fullscreen-label').textContent = workspaceState.fullScreen ? 'Exit full screen' : 'Full screen';
    }
    refreshMapLayout();
  }

  function bindWorkspaceControls() {
    const leftToggle = document.getElementById('toggle-left-panel');
    const rightToggle = document.getElementById('toggle-right-panel');
    const focusToggle = document.getElementById('btn-focus-map');
    const fullScreenToggle = document.getElementById('btn-map-fullscreen');

    if (leftToggle) {
      leftToggle.addEventListener('click', () => {
        workspaceState.leftOpen = !workspaceState.leftOpen;
        applyWorkspaceState();
      });
    }
    if (rightToggle) {
      rightToggle.addEventListener('click', () => {
        workspaceState.rightOpen = !workspaceState.rightOpen;
        applyWorkspaceState();
      });
    }
    if (focusToggle) {
      focusToggle.addEventListener('click', () => {
        if (!workspaceState.focus) {
          workspaceState.previous = {
            leftOpen: workspaceState.leftOpen,
            rightOpen: workspaceState.rightOpen,
          };
          workspaceState.focus = true;
          workspaceState.leftOpen = false;
          workspaceState.rightOpen = false;
        } else {
          const previous = workspaceState.previous || { leftOpen: true, rightOpen: true };
          workspaceState.focus = false;
          workspaceState.leftOpen = previous.leftOpen;
          workspaceState.rightOpen = previous.rightOpen;
          workspaceState.previous = null;
        }
        applyWorkspaceState();
      });
    }
    if (fullScreenToggle) {
      fullScreenToggle.addEventListener('click', () => {
        if (!workspaceState.fullScreen) {
          workspaceState.previous = {
            leftOpen: workspaceState.leftOpen,
            rightOpen: workspaceState.rightOpen,
            focus: workspaceState.focus,
          };
          workspaceState.fullScreen = true;
          workspaceState.focus = true;
          workspaceState.leftOpen = false;
          workspaceState.rightOpen = false;
        } else {
          const previous = workspaceState.previous || { leftOpen: true, rightOpen: true, focus: false };
          workspaceState.fullScreen = false;
          workspaceState.leftOpen = previous.leftOpen;
          workspaceState.rightOpen = previous.rightOpen;
          workspaceState.focus = previous.focus;
          workspaceState.previous = null;
        }
        applyWorkspaceState();
      });
    }
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && workspaceState.fullScreen) {
        fullScreenToggle?.click();
      }
    });
    applyWorkspaceState();
  }

  function updateSarPreviewImages(sceneId) {
    const base = '/data/scenes/' + sceneId;
    const radar = document.getElementById('sar-preview-radar');
    const thermal = document.getElementById('sar-preview-thermal');
    const falseColor = document.getElementById('sar-preview-false');
    if (radar) radar.src = base + '_preview.png';
    if (thermal) thermal.src = base + '_thermal.png';
    if (falseColor) falseColor.src = base + '_false_color.png';
  }

  function bindSarPreviewControls() {
    document.querySelectorAll('.sar-preview').forEach((button) => {
      button.addEventListener('click', () => {
        const palette = button.getAttribute('data-palette');
        const select = document.getElementById('sar-palette-select');
        document.querySelectorAll('.sar-preview').forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        if (select && palette) {
          select.value = palette;
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
    });
  }

  function bindEventHandlers() {
    const btnRun = document.getElementById('btn-run');
    const btnPdf = document.getElementById('btn-export-pdf');
    const sceneSelect = document.getElementById('scene-select');

    if (btnRun) {
      btnRun.addEventListener('click', () => {
        runAttributionPipeline(activeSceneId);
      });
    }

    if (btnPdf) {
      btnPdf.addEventListener('click', () => {
        exportPdf();
      });
    }

    const btnLiveAis = document.getElementById('btn-live-ais');
    if (btnLiveAis) {
      btnLiveAis.addEventListener('click', async () => {
        btnLiveAis.innerHTML = '<span>⏳</span> STREAMING...';
        btnLiveAis.disabled = true;
        try {
          const res = await fetch('/api/ais/live-stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ duration_seconds: 4, max_records: 20 })
          });
          const data = await res.json();
          if (data.status === 'success' && window.MapEngine) {
            MapEngine.renderLiveAIS(data.records);
            console.log(`[TideTrace] Ingested ${data.ingested_count} live vessels from AISStream.io`);
            if (window.UIEngine && UIEngine.appendTrace) {
              UIEngine.appendTrace(`LIVE AIS: Ingested ${data.ingested_count} real-time vessels via AISStream.io`);
            }
          }
        } catch (err) {
          console.warn('[TideTrace] Live AIS fetch error:', err);
        } finally {
          btnLiveAis.textContent = 'LIVE AIS STREAM';
          btnLiveAis.disabled = false;
        }
      });
    }

    const btnLiveMetocean = document.getElementById('btn-live-metocean');
    if (btnLiveMetocean) {
      btnLiveMetocean.addEventListener('click', async () => {
        btnLiveMetocean.innerHTML = '<span>⏳</span> FETCHING...';
        btnLiveMetocean.disabled = true;
        try {
          const res = await fetch('/api/metocean/live-fetch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene_id: activeSceneId })
          });
          const data = await res.json();
          console.log('[TideTrace] Live metocean fetch response:', data);
          if (window.UIEngine && UIEngine.appendTrace) {
            UIEngine.appendTrace(`METOCEAN: Synchronized ${data.provider} (${data.mode})`);
          }
        } catch (err) {
          console.warn('[TideTrace] Live metocean error:', err);
        } finally {
          btnLiveMetocean.textContent = 'LIVE METOCEAN';
          btnLiveMetocean.disabled = false;
        }
      });
    }

    const btnModeLive = document.getElementById('btn-mode-live');
    const btnModeOffline = document.getElementById('btn-mode-offline');
    const badgeText = document.getElementById('airgap-badge-text');

    if (btnModeLive && btnModeOffline) {
      btnModeLive.addEventListener('click', () => {
        isLiveMode = true;
        btnModeLive.classList.add('active', 'live');
        btnModeOffline.classList.remove('active');
        if (badgeText) badgeText.innerText = 'LIVE STREAMING';
        if (window.UIEngine && UIEngine.appendTrace) {
          UIEngine.appendTrace('SYSTEM: Switched primary workflow to LIVE DATA STREAM (AISStream.io + ECMWF ERA5)');
        }
      });

      btnModeOffline.addEventListener('click', () => {
        isLiveMode = false;
        btnModeOffline.classList.add('active');
        btnModeLive.classList.remove('active', 'live');
        if (badgeText) badgeText.innerText = 'OFFLINE FAIL-SAFE';
        if (window.UIEngine && UIEngine.appendTrace) {
          UIEngine.appendTrace('SYSTEM: Reverted to OFFLINE FAIL-SAFE (Pre-Cached Air-Gapped Datasets)');
        }
      });
    }

    if (sceneSelect) {
      sceneSelect.addEventListener('change', (e) => {
        activeSceneId = e.target.value;
        const sc = scenesCatalog.find((s) => s.id === activeSceneId);
        if (sc && window.MapEngine) {
          MapEngine.clearAllLayers();
          MapEngine.renderSAROverlay(sc.id, sc.bounds);
        }
        updateSarPreviewImages(activeSceneId);
        runAttributionPipeline(activeSceneId);
      });
    }
  }

  // ============================================================================
  // 1. API: HEALTH STATUS
  // ============================================================================
  async function fetchHealthStatus() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) return;
      const data = await res.json();
      console.log('[TideTrace] System Status:', data);

      const airgapEl = document.getElementById('airgap-badge');
      if (airgapEl && data.air_gapped_badge) {
        airgapEl.innerHTML = '<div class="pulse-dot"></div><span>OFFLINE · AIR-GAPPED</span>';
      }
    } catch (err) {
      console.warn('[TideTrace] Health check warning:', err);
    }
  }

  // ============================================================================
  // 2. API: SCENES CATALOG
  // ============================================================================
  async function loadScenesAndInit() {
    try {
      const res = await fetch('/api/scenes');
      if (!res.ok) throw new Error('Failed to load scenes');
      const data = await res.json();
      const scenes = data.scenes || [];
      scenesCatalog = scenes;

      const selectEl = document.getElementById('scene-select');
      if (selectEl && scenes.length > 0) {
        selectEl.innerHTML = scenes
          .map((s) => '<option value="' + s.id + '">' + s.title + '</option>')
          .join('');

        activeSceneId = scenes[0].id;
        selectEl.value = activeSceneId;
        updateSarPreviewImages(activeSceneId);
      }

      // Auto-trigger full attribution on default scene
      runAttributionPipeline(activeSceneId);
    } catch (err) {
      console.error('[TideTrace] Error loading scenes:', err);
      // Fallback
      runAttributionPipeline('scene_mumbai_001');
    }
  }

  // ============================================================================
  // 3. API: MASTER RUN ATTRIBUTION PIPELINE
  // ============================================================================
  async function runAttributionPipeline(sceneId) {
    const btnRun = document.getElementById('btn-run');
    if (window.UIEngine && UIEngine.setAnalysisStatus) UIEngine.setAnalysisStatus('Processing', 'neutral');
    if (btnRun) {
      btnRun.disabled = true;
      btnRun.innerHTML = '<span class="pulse-dot" style="background:#fff;"></span> RUNNING...';
    }

    try {
      console.log('[TideTrace] Executing master pipeline for:', sceneId);

      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_id: sceneId,
          hindcast_hours: 48,
          forecast_hours: 36,
          radius_km: 10.0,
          window_hours: 3.0,
          use_live_data: isLiveMode,
        }),
      });

      if (!res.ok) {
        throw new Error('Pipeline error: HTTP ' + res.status);
      }

      const result = await res.json();
      activePipelineResult = result;
      activeJobId = result.job_id;
      console.log('[TideTrace] Pipeline completed successfully:', result);

      // 1. Update Map Layers
      if (window.MapEngine) {
        MapEngine.clearAllLayers();
        const sc = scenesCatalog.find((s) => s.id === sceneId);
        const bounds =
          (result.detection &&
            result.detection.properties &&
            result.detection.properties.bounds) ||
          (sc ? sc.bounds : [18.75, 71.95, 19.15, 72.35]);
        MapEngine.renderSAROverlay(sceneId, bounds);
        MapEngine.renderDetection(result.detection);
        MapEngine.renderDrift(result.drift);
        MapEngine.renderAIS(result.attribution ? result.attribution.suspects : []);
      }

      // 2. Update UI Panels
      if (window.UIEngine) {
        UIEngine.renderLeaderboard(result.attribution ? result.attribution.suspects : []);
        UIEngine.renderTelemetry(result.detection, result.drift);
        UIEngine.renderExecutionTrace(result.execution_trace, result.total_duration_ms);
        UIEngine.renderSummary(result, scenesCatalog.find((s) => s.id === sceneId));
      }

      // 3. Configure Timeline Scrubber Range
      if (window.TimelineEngine && result.drift) {
        const hind = result.drift.hindcast || {};
        const fore = result.drift.forecast || {};

        const tSat = result.t_sat ? new Date(result.t_sat).getTime() / 1000 : Date.now() / 1000;
        const tOrigin = hind.origin_timestamp || (tSat - 24 * 3600);
        const tForeEnd = fore.forecast_end_time_iso ? new Date(fore.forecast_end_time_iso).getTime() / 1000 : (tSat + 36 * 3600);

        TimelineEngine.configureRange(tOrigin, tSat, tForeEnd);
      }

    } catch (err) {
      console.error('[TideTrace] Pipeline execution failed:', err);
      if (window.UIEngine) {
        if (UIEngine.setAnalysisStatus) UIEngine.setAnalysisStatus('Analysis failed', 'alert');
        UIEngine.renderExecutionTrace([
          { step: 'ERROR', name: 'Pipeline Execution Error', duration_ms: 0, status: 'skipped', details: err.message }
        ], 0);
      }
    } finally {
      if (btnRun) {
        btnRun.disabled = false;
        btnRun.textContent = 'RUN ATTRIBUTION';
      }
    }
  }

  // ============================================================================
  // 4. API: EXPORT PDF DOSSIER
  // ============================================================================
  function exportPdf(optionalId) {
    const targetId = optionalId || activeJobId || activeSceneId || 'scene_mumbai_001';
    console.log('[TideTrace] Downloading PDF Dossier for:', targetId);
    window.open('/api/report/' + targetId, '_blank');
  }

  const engine = {
    init,
    runAttributionPipeline,
    exportPdf,
  };
  window.AppEngine = engine;
  return engine;
})();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  AppEngine.init();
});
