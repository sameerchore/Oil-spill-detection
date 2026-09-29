/**
 * TideTrace Synchronized Timeline Engine
 * Controls synchronized temporal scrubbing between Origin Time [T_origin],
 * Satellite Detection [T_sat], and Forward Forecast Horizon [T_sat + 36h].
 */

const TimelineEngine = (function () {
  let isPlaying = false;
  let animationInterval = null;
  let playbackSpeed = 5; // Multiplier

  let startEpoch = 0; // T_origin
  let satEpoch = 0;   // T_sat
  let endEpoch = 0;   // T_sat + 36h
  let currentEpoch = 0;

  // DOM Elements
  let scrubber = null;
  let hudTime = null;
  let playBtn = null;
  let speedSelect = null;
  let lblStart = null;
  let lblSat = null;
  let lblEnd = null;

  function initTimeline() {
    scrubber = document.getElementById('time-scrubber');
    hudTime = document.getElementById('current-time-hud');
    playBtn = document.getElementById('btn-play-pause');
    speedSelect = document.getElementById('select-speed');
    lblStart = document.getElementById('lbl-timeline-start');
    lblSat = document.getElementById('lbl-timeline-sat');
    lblEnd = document.getElementById('lbl-timeline-end');

    if (!scrubber) return;

    // Scrubber drag / input event
    scrubber.addEventListener('input', (e) => {
      const pct = parseFloat(e.target.value) / 100.0;
      currentEpoch = startEpoch + pct * (endEpoch - startEpoch);
      updateDisplay();
    });

    // Play/Pause button
    if (playBtn) {
      playBtn.addEventListener('click', togglePlay);
    }

    // Controls
    const btnStepBack = document.getElementById('btn-step-back');
    const btnStepFwd = document.getElementById('btn-step-fwd');
    const btnResetOrig = document.getElementById('btn-reset-orig');
    const btnResetSat = document.getElementById('btn-reset-sat');

    if (btnStepBack) {
      btnStepBack.addEventListener('click', () => stepTime(-3600));
    }
    if (btnStepFwd) {
      btnStepFwd.addEventListener('click', () => stepTime(3600));
    }
    if (btnResetOrig) {
      btnResetOrig.addEventListener('click', () => setTime(startEpoch));
    }
    if (btnResetSat) {
      btnResetSat.addEventListener('click', () => setTime(satEpoch));
    }

    // Speed selector
    if (speedSelect) {
      speedSelect.addEventListener('change', (e) => {
        playbackSpeed = parseFloat(e.target.value) || 5;
        if (isPlaying) {
          stopAnimation();
          startAnimation();
        }
      });
    }

    // Default 24h baseline if not loaded
    const now = Math.floor(Date.now() / 1000);
    configureRange(now - 86400, now, now + 36 * 3600);
  }

  function configureRange(tOriginEpoch, tSatEpoch, tForecastEndEpoch) {
    startEpoch = tOriginEpoch;
    satEpoch = tSatEpoch;
    endEpoch = tForecastEndEpoch;
    currentEpoch = satEpoch; // Default cursor at detection instant

    if (lblStart) {
      const dStart = new Date(startEpoch * 1000);
      lblStart.textContent = 'T_orig: ' + dStart.toISOString().substring(11, 16) + 'Z';
    }
    if (lblSat) {
      const dSat = new Date(satEpoch * 1000);
      lblSat.textContent = 'T_sat: ' + dSat.toISOString().substring(11, 16) + 'Z';
    }
    if (lblEnd) {
      const dEnd = new Date(endEpoch * 1000);
      lblEnd.textContent = '+36h: ' + dEnd.toISOString().substring(11, 16) + 'Z';
    }

    updateDisplay();
  }

  function togglePlay() {
    if (isPlaying) {
      stopAnimation();
    } else {
      startAnimation();
    }
  }

  function startAnimation() {
    if (isPlaying) return;
    isPlaying = true;
    if (playBtn) playBtn.innerHTML = '&#10074;&#10074;'; // Pause icon

    const tickIntervalMs = 80;
    // Advance simulated time per tick (e.g. 5x speed = 300s / tick)
    const simulatedStepSeconds = (tickIntervalMs / 1000.0) * (3600.0 * (playbackSpeed / 2.0));

    animationInterval = setInterval(() => {
      currentEpoch += simulatedStepSeconds;
      if (currentEpoch > endEpoch) {
        currentEpoch = startEpoch; // Loop
      }
      updateDisplay();
    }, tickIntervalMs);
  }

  function stopAnimation() {
    if (!isPlaying) return;
    isPlaying = false;
    if (playBtn) playBtn.innerHTML = '&#9654;'; // Play icon
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }
  }

  function stepTime(deltaSeconds) {
    currentEpoch = Math.max(startEpoch, Math.min(endEpoch, currentEpoch + deltaSeconds));
    updateDisplay();
  }

  function setTime(epoch) {
    currentEpoch = Math.max(startEpoch, Math.min(endEpoch, epoch));
    updateDisplay();
  }

  function updateDisplay() {
    if (!hudTime || !scrubber) return;

    // Update Slider percentage
    const totalSpan = Math.max(endEpoch - startEpoch, 1);
    const pct = ((currentEpoch - startEpoch) / totalSpan) * 100.0;
    scrubber.value = pct;

    // Format ISO string
    const d = new Date(currentEpoch * 1000);
    const isoString = d.toISOString().replace('.000Z', 'Z');
    hudTime.textContent = isoString;

    // Notify MapEngine to interpolate active positions
    if (window.MapEngine && MapEngine.updateTimePositions) {
      MapEngine.updateTimePositions(currentEpoch);
    }
  }

  const engine = {
    initTimeline,
    configureRange,
    setTime,
    togglePlay,
  };
  window.TimelineEngine = engine;
  return engine;
})();

