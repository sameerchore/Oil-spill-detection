/**
 * TideTrace Tactical UI & Inspector Engine
 * Manages Suspect Leaderboard, Dark Vessel Anomaly Inspector, Telemetry KPIs,
 * Step-by-Step Execution Trace Logs, and Forensic Detail Modals.
 */

const UIEngine = (function () {

  function initUI() {
    setupTabSwitchers();
    setupModalHandlers();
    startLiveClock();
  }

  function startLiveClock() {
    const clockEl = document.getElementById('utc-clock');
    if (!clockEl) return;
    setInterval(() => {
      const now = new Date();
      clockEl.textContent = now.toISOString().substring(11, 19) + ' UTC';
    }, 1000);
  }

  function setupTabSwitchers() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        tabBtns.forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach((c) => (c.style.display = 'none'));

        btn.classList.add('active');
        const targetTab = btn.getAttribute('data-tab');
        const targetEl = document.getElementById(targetTab);
        if (targetEl) targetEl.style.display = 'block';
      });
    });
  }

  function setupModalHandlers() {
    const modal = document.getElementById('vessel-modal');
    const closeBtn = document.getElementById('modal-close-btn');
    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => modal.classList.remove('open'));
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('open');
      });
    }
  }

  // ============================================================================
  // 1. RENDER SUSPECT LEADERBOARD
  // ============================================================================
  function renderLeaderboard(suspects) {
    const listEl = document.getElementById('suspect-list');
    const badgeEl = document.getElementById('suspect-count-badge');
    const darkListEl = document.getElementById('dark-vessel-list');
    if (!listEl) return;

    if (!suspects || !suspects.length) {
      listEl.innerHTML = '<div style="padding:24px; text-align:center; color:#57677d; font-size:10.5px;"><b>NO SUSPECT VESSELS IDENTIFIED</b><br/>Zero vessels found within origin search radius.<br/>Clean ocean true-negative scenario verified.</div>';
      if (badgeEl) badgeEl.textContent = '0 SUSPECTS';
      if (darkListEl) darkListEl.innerHTML = '<div style="padding:16px; color:#57677d; font-size:10px;">No dark vessel gaps detected.</div>';
      return;
    }

    if (badgeEl) badgeEl.textContent = suspects.length + ' EVALUATED';

    let html = '';
    let darkHtml = '';

    suspects.forEach((s) => {
      const isCulprit = s.rank === 1;
      const rankClass = isCulprit ? 'rank-1' : (s.rank === 2 ? 'rank-2' : '');
      const cardClass = isCulprit ? 'suspect-card culprit' : 'suspect-card';

      const proxPct = Math.round((s.s_prox || 0) * 100);
      const typePct = Math.round((s.s_type || 0) * 100);
      const trajPct = Math.round((s.s_traj || 0) * 100);
      const behPct = Math.round((s.s_beh || 0) * 100);

      const flagsHtml = (s.anomaly_flags || [])
        .map((f) => '<span class="anomaly-pill">' + f + '</span>')
        .join('');

      html += '<div class="' + cardClass + '" data-mmsi="' + s.mmsi + '">' +
        '<div class="suspect-header">' +
          '<div style="display:flex; align-items:center; gap:6px;">' +
            '<span class="suspect-rank-badge ' + rankClass + '">#' + s.rank + '</span>' +
            '<span class="suspect-name">' + s.vessel_name + '</span>' +
          '</div>' +
          '<span style="font-size:11px; font-weight:bold; color:' + (isCulprit ? '#D65C5C' : '#D6A84F') + ';">' +
            (s.confidence_pct || 0).toFixed(1) + '%' +
          '</span>' +
        '</div>' +
        '<div class="suspect-meta">' +
          '<span>MMSI: ' + s.mmsi + ' &middot; ' + (s.vessel_type || 'Cargo') + '</span>' +
          '<span>CPA: <b>' + (s.distance_km || 0).toFixed(2) + ' km</b></span>' +
        '</div>' +
        '<div class="score-breakdown-grid">' +
          '<div>' +
            '<span>PROX (40%): ' + proxPct + '%</span>' +
            '<div class="score-bar-container"><div class="score-bar-fill red" style="width:' + proxPct + '%"></div></div>' +
          '</div>' +
          '<div>' +
            '<span>TYPE (20%): ' + typePct + '%</span>' +
            '<div class="score-bar-container"><div class="score-bar-fill yellow" style="width:' + typePct + '%"></div></div>' +
          '</div>' +
          '<div>' +
            '<span>TRAJ (15%): ' + trajPct + '%</span>' +
            '<div class="score-bar-container"><div class="score-bar-fill" style="width:' + trajPct + '%"></div></div>' +
          '</div>' +
          '<div>' +
            '<span>BEH (25%): ' + behPct + '%</span>' +
            '<div class="score-bar-container"><div class="score-bar-fill green" style="width:' + behPct + '%"></div></div>' +
          '</div>' +
        '</div>' +
        (flagsHtml ? '<div style="margin-top:6px;">' + flagsHtml + '</div>' : '') +
      '</div>';

      if (s.s_beh >= 0.5 || (s.anomaly_flags && s.anomaly_flags.length)) {
        darkHtml += '<div class="suspect-card culprit" style="cursor:pointer;" data-mmsi="' + s.mmsi + '">' +
          '<div class="suspect-header">' +
            '<span class="suspect-name" style="color:#D65C5C;">' + s.vessel_name + '</span>' +
            '<span style="font-size:9.5px; color:#8b9bb4;">MMSI: ' + s.mmsi + '</span>' +
          '</div>' +
          '<div style="font-size:9px; color:#8b9bb4; margin-bottom:4px;">' +
            '<b>Anomaly Score S_beh:</b> ' + (s.s_beh || 0).toFixed(3) +
          '</div>' +
          flagsHtml +
        '</div>';
      }
    });

    listEl.innerHTML = html;
    if (darkListEl) {
      darkListEl.innerHTML = darkHtml || '<div style="padding:16px; color:#57677d; font-size:10px;">No dark vessel gaps detected in origin window.</div>';
    }

    document.querySelectorAll('.suspect-card').forEach((card) => {
      card.addEventListener('click', () => {
        const mmsi = parseInt(card.getAttribute('data-mmsi'));
        const suspect = suspects.find((x) => x.mmsi === mmsi);
        if (suspect) {
          document.querySelectorAll('.suspect-card').forEach((c) => c.classList.remove('active'));
          card.classList.add('active');

          if (suspect.cpa_lat && suspect.cpa_lon && window.MapEngine) {
            MapEngine.focusOnVessel(suspect.mmsi, suspect.cpa_lat, suspect.cpa_lon);
          }
          openVesselModal(suspect);
        }
      });
    });
  }

  // ============================================================================
  // 2. RENDER TELEMETRY & METOCEAN KPIS
  // ============================================================================
  function renderTelemetry(detectionFC, driftData) {
    if (!detectionFC) return;

    const features = detectionFC.features || [];
    const oilFeature = features.find((f) => f.properties && (f.properties.class_id === 2 || f.properties.class_name === 'mineral_oil'));
    const props = oilFeature ? oilFeature.properties : (features[0] && features[0].properties ? features[0].properties : {});

    const elArea = document.getElementById('tel-area');
    const elPerim = document.getElementById('tel-perimeter');
    const elLen = document.getElementById('tel-length');
    const elWid = document.getElementById('tel-width');
    const elOrient = document.getElementById('tel-orientation');
    const elComp = document.getElementById('tel-compactness');

    if (elArea) elArea.textContent = props.area_km2 !== undefined ? props.area_km2.toFixed(2) + ' km²' : '--';
    if (elPerim) elPerim.textContent = props.perimeter_km !== undefined ? props.perimeter_km.toFixed(2) + ' km' : '--';
    if (elLen) elLen.textContent = props.length_km !== undefined ? props.length_km.toFixed(2) + ' km' : '--';
    if (elWid) elWid.textContent = props.width_km !== undefined ? props.width_km.toFixed(2) + ' km' : '--';
    if (elOrient) elOrient.textContent = props.orientation_deg !== undefined ? props.orientation_deg.toFixed(1) + '°' : '--';
    if (elComp) elComp.textContent = props.compactness !== undefined ? props.compactness.toFixed(3) : '--';

    if (driftData) {
      const hindcast = driftData.hindcast || {};
      const forecast = driftData.forecast || {};

      const elOrigTime = document.getElementById('tel-origin-time');
      const elDriftAge = document.getElementById('tel-drift-age');
      const elHindDist = document.getElementById('tel-hindcast-drift');
      const elForeDist = document.getElementById('tel-forecast-drift');

      if (elOrigTime) elOrigTime.textContent = hindcast.origin_time_iso || '--';
      if (elDriftAge) elDriftAge.textContent = hindcast.age_hours_proxy ? hindcast.age_hours_proxy + ' h' : '--';
      if (elHindDist) elHindDist.textContent = hindcast.total_drift_km ? hindcast.total_drift_km.toFixed(1) + ' km' : '--';
      if (elForeDist) elForeDist.textContent = forecast.total_drift_km ? forecast.total_drift_km.toFixed(1) + ' km' : '--';
    }
  }

  function setAnalysisStatus(label, tone) {
    const statusEl = document.getElementById('summary-status');
    if (!statusEl) return;
    statusEl.textContent = label;
    statusEl.classList.remove('status-neutral', 'status-good', 'status-alert');
    statusEl.classList.add('status-' + (tone || 'neutral'));
  }

  function renderSummary(result, scene) {
    const detection = result && result.detection ? result.detection : {};
    const properties = detection.properties || {};
    const features = detection.features || [];
    const oilFeatures = features.filter((feature) => {
      const props = feature.properties || {};
      return props.class_id === 2 || props.class_name === 'mineral_oil';
    });
    const suspects = result && result.attribution && result.attribution.suspects
      ? result.attribution.suspects : [];
    const confidence = oilFeatures.length
      ? Math.max(...oilFeatures.map((feature) => Number(feature.properties.confidence || 0))) * 100
      : 0;
    const area = properties.total_oil_area_km2 !== undefined
      ? Number(properties.total_oil_area_km2)
      : oilFeatures.reduce((sum, feature) => sum + Number(feature.properties.area_km2 || 0), 0);
    const coords = (oilFeatures[0] && oilFeatures[0].properties && oilFeatures[0].properties.centroid)
      || (scene && scene.centroid);

    const confidenceEl = document.getElementById('summary-confidence');
    const areaEl = document.getElementById('summary-area');
    const vesselsEl = document.getElementById('summary-vessels');
    const coordsEl = document.getElementById('summary-coords');
    const timeEl = document.getElementById('summary-time');

    if (confidenceEl) confidenceEl.textContent = oilFeatures.length ? confidence.toFixed(1) + '%' : '—';
    if (areaEl) areaEl.textContent = oilFeatures.length ? area.toFixed(2) + ' km²' : '0.00 km²';
    if (vesselsEl) vesselsEl.textContent = suspects.length ? String(suspects.length) : '0';
    if (coordsEl && coords && coords.length >= 2) {
      coordsEl.textContent = Number(coords[0]).toFixed(4) + '°N, ' + Number(coords[1]).toFixed(4) + '°E';
    }
    if (timeEl) timeEl.textContent = result && result.total_duration_ms !== undefined
      ? Number(result.total_duration_ms).toFixed(0) + ' ms' : '—';

    setAnalysisStatus(oilFeatures.length ? 'Oil spill detected' : 'No spill detected', oilFeatures.length ? 'alert' : 'good');
  }

  // ============================================================================
  // 3. RENDER EXECUTION TRACE LOGS
  // ============================================================================
  function renderExecutionTrace(traceLogs, totalDurationMs) {
    const container = document.getElementById('trace-log-container');
    const totalBadge = document.getElementById('total-time-badge');
    if (!container) return;

    if (totalBadge && totalDurationMs !== undefined) {
      totalBadge.textContent = totalDurationMs.toFixed(1) + ' ms';
    }

    if (!traceLogs || !traceLogs.length) {
      container.innerHTML = '<div style="color:#57677d; font-size:9px;">Execution trace pending...</div>';
      return;
    }

    let html = '';
    traceLogs.forEach((log) => {
      const statusClass = log.status === 'completed' ? 'completed' : 'skipped';
      const stateLabel = log.status === 'completed' ? 'COMPLETE' : 'PENDING';

      html += '<div class="trace-log-item ' + statusClass + '">' +
        '<div class="trace-log-header">' +
          '<span>' + stateLabel + ' · ' + (log.name || log.step) + '</span>' +
          '<span class="trace-log-duration">' + (log.duration_ms || 0).toFixed(1) + ' ms</span>' +
        '</div>' +
        '<div class="trace-log-details">' + (log.details || '') + '</div>' +
      '</div>';
    });

    container.innerHTML = html;
  }

  // ============================================================================
  // 4. OPEN VESSEL DEEP-DIVE MODAL
  // ============================================================================
  function openVesselModal(suspect) {
    const modal = document.getElementById('vessel-modal');
    const titleEl = document.getElementById('modal-vessel-title');
    const bodyEl = document.getElementById('modal-vessel-body');
    if (!modal || !bodyEl) return;

    const isCulprit = suspect.rank === 1;
    if (titleEl) {
      titleEl.innerHTML = 'VESSEL FORENSIC DOSSIER &middot; <span style="color:' + (isCulprit ? '#D65C5C' : '#D6A84F') + ';">' + suspect.vessel_name + '</span> [#' + suspect.rank + ']';
    }

    const reasonsHtml = (suspect.reasons || [])
      .map((r) => '<li style="margin-bottom:4px;">' + r + '</li>')
      .join('');

    bodyEl.innerHTML = '<div style="font-family:var(--font-mono); font-size:11px; line-height:1.5;">' +
      '<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px; background:#0c131f; padding:12px; border-radius:6px; border:1px solid rgba(0,240,255,0.3);">' +
        '<div>' +
          '<div><b>MMSI:</b> ' + suspect.mmsi + '</div>' +
          '<div><b>Vessel Type:</b> ' + (suspect.vessel_type || 'Cargo') + '</div>' +
          '<div><b>CPA Distance:</b> <font color="#D65C5C"><b>' + (suspect.distance_km || 0).toFixed(2) + ' km</b></font></div>' +
          '<div><b>CPA Timestamp:</b> ' + suspect.closest_time + '</div>' +
        '</div>' +
        '<div>' +
          '<div><b>Speed at CPA:</b> ' + (suspect.cpa_sog || 0) + ' knots</div>' +
          '<div><b>Course at CPA:</b> ' + (suspect.cpa_cog || 0) + '&deg;</div>' +
          '<div><b>Total Score S:</b> ' + (suspect.s_total || 0).toFixed(3) + '</div>' +
          '<div><b>Confidence %:</b> <font color="' + (isCulprit ? '#D65C5C' : '#4FA89A') + '"><b>' + (suspect.confidence_pct || 0).toFixed(1) + '%</b></font></div>' +
        '</div>' +
      '</div>' +
      '<div style="font-size:11px; font-weight:bold; color:#D6A84F; margin-bottom:6px;">' +
        'PROSECUTION GROUNDS &amp; MARPOL ANNEX I EVIDENCE:' +
      '</div>' +
      '<ul style="padding-left:18px; color:#f0f6fc; margin-bottom:14px;">' +
        reasonsHtml +
      '</ul>' +
      '<div style="text-align:right;">' +
        '<button class="btn-tactical-primary" onclick="if (window.AppEngine) AppEngine.exportPdf();">' +
          'EXPORT FULL LEGAL DOSSIER (PDF)' +
        '</button>' +
      '</div>' +
    '</div>';

    modal.classList.add('open');
  }

  const engine = {
    initUI,
    renderLeaderboard,
    renderTelemetry,
    renderSummary,
    setAnalysisStatus,
    renderExecutionTrace,
    openVesselModal,
  };
  window.UIEngine = engine;
  return engine;
})();
