/**
 * main.js
 * NeuroVoice AI — Clinical Dysarthria Screening Platform
 * Handles:
 *   - Knowledge hub tab switching
 *   - Drag-and-drop + file browser upload
 *   - Live microphone recording with canvas waveform visualizer
 *   - Preset audio sample loading and auto-analysis
 *   - AJAX form submission + step-by-step status feedback
 *   - Animated result rendering with spectrogram heatmap
 *   - Billing toggle (monthly / annual)
 *   - Pricing plan modal
 *   - Print / copy JSON report actions
 */

document.addEventListener("DOMContentLoaded", () => {

  /* -----------------------------------------------------------------------
     1. KNOWLEDGE HUB TABS
  ----------------------------------------------------------------------- */
  const tabNavBtns = document.querySelectorAll(".tab-nav-btn");
  const tabPanels  = document.querySelectorAll(".tab-content-panel");

  tabNavBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tabTarget;
      tabNavBtns.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(target);
      if (panel) panel.classList.add("active");
    });
  });

  /* -----------------------------------------------------------------------
     2. MAIN SCREENING DASHBOARD ELEMENTS
  ----------------------------------------------------------------------- */
  const dropZone          = document.getElementById("dropZone");
  const fileInput         = document.getElementById("audioFileInput");
  const recordBtn         = document.getElementById("recordBtn");
  const recordBtnText     = document.getElementById("recordBtnText");
  const analyzeForm       = document.getElementById("analyzeForm");
  const submitBtn         = document.getElementById("submitBtn");
  const loadingSpinner    = document.getElementById("loadingSpinner");
  const selectedSampleInput = document.getElementById("selectedSampleName");

  const filePreviewArea   = document.getElementById("filePreviewArea");
  const dropZonePrompt    = document.getElementById("dropZonePrompt");
  const previewFilename   = document.getElementById("previewFilename");
  const previewFilesize   = document.getElementById("previewFilesize");
  const audioPlayer       = document.getElementById("audioPlayer");
  const clearFileBtn      = document.getElementById("clearFileBtn");

  const processingStatus  = document.getElementById("processingStatus");
  const statusStepText    = document.getElementById("statusStepText");
  const progressBarFill   = document.getElementById("progressBarFill");
  const errorAlert        = document.getElementById("errorAlert");
  const errorMessage      = document.getElementById("errorMessage");

  const resultsPlaceholder     = document.getElementById("resultsPlaceholder");
  const resultsContainer       = document.getElementById("resultsContainer");
  const resultHeroCard         = document.getElementById("resultHeroCard");
  const triageStatusPill       = document.getElementById("triageStatusPill");
  const triageDot              = document.getElementById("triageDot");
  const classificationTag      = document.getElementById("classificationTag");
  const predictionTitle        = document.getElementById("predictionTitle");
  const modelScoreVal          = document.getElementById("modelScoreVal");
  const scoreHealthyText       = document.getElementById("scoreHealthyText");
  const scoreDysarthricText    = document.getElementById("scoreDysarthricText");
  const probFillHealthy        = document.getElementById("probFillHealthy");
  const probFillDysarthric     = document.getElementById("probFillDysarthric");
  const summaryNote            = document.getElementById("summaryNote");
  const resultWarnings         = document.getElementById("resultWarnings");
  const signalMetricsRow       = document.getElementById("signalMetricsRow");
  const spectrogramCanvas      = document.getElementById("spectrogramCanvas");
  const analysisTimestamp      = document.getElementById("analysisTimestamp");
  const clinicalRecommendationBox = document.getElementById("clinicalRecommendationBox");
  const recTitle               = document.getElementById("recTitle");
  const recDesc                = document.getElementById("recDesc");
  const liveVisualizerContainer = document.getElementById("liveVisualizerContainer");
  const liveAudioCanvas        = document.getElementById("liveAudioCanvas");
  const recordTimerDisplay     = document.getElementById("recordTimerDisplay");

  /* -----------------------------------------------------------------------
     3. MICROPHONE RECORDING STATE
  ----------------------------------------------------------------------- */
  let mediaRecorder     = null;
  let audioChunks       = [];
  let isRecording       = false;
  let recordTimer       = null;
  let recordSeconds     = 0;
  let recordedBlob      = null;
  let liveAnimationFrame = null;
  let audioContext      = null;
  let analyserNode      = null;

  /* -----------------------------------------------------------------------
     4. DRAG AND DROP
  ----------------------------------------------------------------------- */
  if (dropZone) {
    ["dragenter", "dragover"].forEach(evt => {
      dropZone.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(evt => {
      dropZone.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });
    dropZone.addEventListener("drop", e => {
      const files = e.dataTransfer.files;
      if (files.length > 0) handleFileSelect(files[0]);
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", e => {
      if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });
  }

  if (clearFileBtn) {
    clearFileBtn.addEventListener("click", () => clearSelectedAudio());
  }

  function handleFileSelect(file) {
    hideError();
    selectedSampleInput.value = "";
    recordedBlob = null;
    stopLiveVisualizer();

    previewFilename.textContent = file.name;
    previewFilesize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB — ${file.type || "audio"}`;

    const fileUrl = URL.createObjectURL(file);
    audioPlayer.src = fileUrl;

    dropZonePrompt.style.display = "none";
    filePreviewArea.style.display = "block";
    if (liveVisualizerContainer) liveVisualizerContainer.style.display = "none";
    submitBtn.disabled = false;
  }

  function clearSelectedAudio() {
    fileInput.value = "";
    selectedSampleInput.value = "";
    recordedBlob = null;
    audioPlayer.src = "";
    filePreviewArea.style.display = "none";
    dropZonePrompt.style.display = "block";
    submitBtn.disabled = true;
    if (liveVisualizerContainer) liveVisualizerContainer.style.display = "none";
    stopLiveVisualizer();
    hideError();
  }

  /* -----------------------------------------------------------------------
     5. PRESET SAMPLE LOADER (auto-analyze on click)
  ----------------------------------------------------------------------- */
  window.loadPresetSample = function(filename, label) {
    hideError();
    fileInput.value = "";
    recordedBlob = null;
    selectedSampleInput.value = filename;

    const shortName = filename.replace("_speech_sample_", " ").replace(/_/g, " ").replace(".wav", "");
    previewFilename.textContent = label;
    previewFilesize.textContent = "Verified clinical demonstration recording";

    audioPlayer.src = `/sample-audio/${encodeURIComponent(filename)}`;

    dropZonePrompt.style.display = "none";
    filePreviewArea.style.display = "block";
    submitBtn.disabled = false;

    // Automatically trigger analysis
    setTimeout(() => submitBtn.click(), 120);
  };

  document.querySelectorAll("[data-sample-name]").forEach(button => {
    button.addEventListener("click", () => {
      window.loadPresetSample(button.dataset.sampleName, button.dataset.sampleLabel);
    });
  });

  /* -----------------------------------------------------------------------
     6. LIVE MICROPHONE RECORDING + CANVAS WAVEFORM
  ----------------------------------------------------------------------- */
  if (recordBtn) {
    recordBtn.addEventListener("click", async () => {
      if (!isRecording) await startRecording();
      else stopRecording();
    });
  }

  async function startRecording() {
    hideError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Microphone access is not supported by your browser. Try Chrome or Edge.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
        const audioUrl = URL.createObjectURL(recordedBlob);
        audioPlayer.src = audioUrl;
        previewFilename.textContent = "live_microphone_recording.webm";
        previewFilesize.textContent = `${(recordedBlob.size / 1024).toFixed(1)} KB — Live Microphone Capture`;
        dropZonePrompt.style.display = "none";
        filePreviewArea.style.display = "block";
        if (liveVisualizerContainer) liveVisualizerContainer.style.display = "none";
        submitBtn.disabled = false;
        stopLiveVisualizer();
      };

      mediaRecorder.start(100);
      isRecording = true;
      recordBtn.classList.add("recording-active");
      recordSeconds = 0;
      recordBtnText.textContent = "Recording… Click to Stop";

      // Show the live canvas visualizer
      if (liveVisualizerContainer) liveVisualizerContainer.style.display = "block";

      // Live waveform canvas
      if (liveAudioCanvas) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 512;
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyserNode);
        drawLiveWaveform();
      }

      recordTimer = setInterval(() => {
        recordSeconds++;
        const mm = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
        const ss = String(recordSeconds % 60).padStart(2, "0");
        if (recordTimerDisplay) recordTimerDisplay.textContent = `${mm}:${ss}`;
        recordBtnText.textContent = `Recording ${mm}:${ss} — Click to Stop`;
        if (recordSeconds >= 15) stopRecording();
      }, 1000);

    } catch (err) {
      showError("Could not access microphone: " + err.message);
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
    isRecording = false;
    clearInterval(recordTimer);
    recordBtn.classList.remove("recording-active");
    recordBtnText.textContent = "Record Live Microphone";
  }

  function drawLiveWaveform() {
    if (!analyserNode || !liveAudioCanvas) return;
    const ctx = liveAudioCanvas.getContext("2d");
    const w = liveAudioCanvas.width;
    const h = liveAudioCanvas.height;
    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);

    function draw() {
      liveAnimationFrame = requestAnimationFrame(draw);
      analyserNode.getByteTimeDomainData(dataArray);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, w, h);

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#f43f5e";
      ctx.beginPath();
      const sliceW = w / dataArray.length;
      let x = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * h / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceW;
      }
      ctx.lineTo(w, h / 2);
      ctx.stroke();
    }
    draw();
  }

  function stopLiveVisualizer() {
    if (liveAnimationFrame) {
      cancelAnimationFrame(liveAnimationFrame);
      liveAnimationFrame = null;
    }
    if (audioContext && audioContext.state !== "closed") {
      audioContext.close().catch(() => {});
      audioContext = null;
      analyserNode = null;
    }
  }

  /* -----------------------------------------------------------------------
     7. FORM SUBMISSION & AJAX INFERENCE
  ----------------------------------------------------------------------- */
  if (analyzeForm) {
    analyzeForm.addEventListener("submit", async e => {
      e.preventDefault();
      hideError();

      const formData = new FormData(analyzeForm);

      // Attach live recorded blob if present
      if (recordedBlob) {
        formData.set("audio_file", recordedBlob, "microphone_recording.webm");
      }

      // UI: start loading state
      submitBtn.disabled = true;
      loadingSpinner.style.display = "inline-block";
      processingStatus.style.display = "block";

      const steps = [
        { msg: "Ingesting audio stream and validating signal quality…", prog: 15 },
        { msg: "Resampling to 16 kHz mono and removing DC offset…", prog: 30 },
        { msg: "Extracting 144-dimensional acoustic biomarker vector…", prog: 55 },
        { msg: "Applying zero-leakage StandardScaler normalization…", prog: 75 },
        { msg: "Running clinical classification engine…", prog: 90 },
        { msg: "Generating Log-Mel spectrogram visualization…", prog: 98 },
      ];

      let stepIdx = 0;
      function advanceStep() {
        if (stepIdx < steps.length) {
          statusStepText.textContent = steps[stepIdx].msg;
          if (progressBarFill) progressBarFill.style.width = steps[stepIdx].prog + "%";
          stepIdx++;
        }
      }
      advanceStep();
      const stepInterval = setInterval(advanceStep, 400);

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json"
          },
          body: formData
        });

        clearInterval(stepInterval);
        if (progressBarFill) progressBarFill.style.width = "100%";

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || "Analysis failed. Please try again.");
        }

        renderResults(data);

      } catch (err) {
        showError(err.message);
      } finally {
        clearInterval(stepInterval);
        submitBtn.disabled = false;
        loadingSpinner.style.display = "none";
        setTimeout(() => {
          processingStatus.style.display = "none";
          if (progressBarFill) progressBarFill.style.width = "0%";
        }, 400);
      }
    });
  }

  /* -----------------------------------------------------------------------
     8. ANIMATED RESULT RENDERING
  ----------------------------------------------------------------------- */
  let lastResult = null;

  function renderResults(res) {
    lastResult = res;
    resultsPlaceholder.style.display = "none";
    resultsContainer.style.display = "block";

    const isHealthy = res.predicted_target === 0;

    // Hero card class
    if (resultHeroCard) {
      resultHeroCard.className = `card result-hero-card ${isHealthy ? "healthy-result" : "dysarthric-result"}`;
    }

    // Triage status pill
    if (triageStatusPill) {
      triageStatusPill.className = `triage-status-pill ${isHealthy ? "status-healthy" : "status-dysarthric"}`;
    }
    if (classificationTag) classificationTag.textContent = isHealthy ? "Acoustic Screening: Normal Phonation" : "Acoustic Screening: Pathological Markers Detected";

    // Prediction title with smooth reveal
    if (predictionTitle) {
      predictionTitle.style.opacity = "0";
      predictionTitle.textContent = res.prediction;
      predictionTitle.style.color = isHealthy ? "#34d399" : "#fbbf24";
      setTimeout(() => { predictionTitle.style.transition = "opacity 0.4s"; predictionTitle.style.opacity = "1"; }, 50);
    }

    // Timestamp
    if (analysisTimestamp) {
      analysisTimestamp.textContent = "Just now — " + new Date().toLocaleTimeString();
    }

    // Score values
    if (modelScoreVal) modelScoreVal.textContent = `${res.model_score_pct}%`;
    if (scoreHealthyText) scoreHealthyText.textContent = `${res.score_healthy_pct}%`;
    if (scoreDysarthricText) scoreDysarthricText.textContent = `${res.score_dysarthric_pct}%`;

    // Animated probability bars
    if (probFillHealthy && probFillDysarthric) {
      probFillHealthy.style.width = "0%";
      probFillDysarthric.style.width = "0%";
      setTimeout(() => {
        probFillHealthy.style.width = `${res.score_healthy_pct}%`;
        probFillDysarthric.style.width = `${res.score_dysarthric_pct}%`;
      }, 100);
    }

    if (summaryNote) summaryNote.textContent = res.summary_note;

    // Warnings
    if (resultWarnings) {
      if (res.warnings && res.warnings.length > 0) {
        resultWarnings.style.display = "block";
        resultWarnings.innerHTML = `
          <strong>Signal Quality Advisory:</strong>
          <ul>${res.warnings.map(w => `<li>${w}</li>`).join("")}</ul>`;
      } else {
        resultWarnings.style.display = "none";
      }
    }

    // Clinical Recommendation
    if (recTitle && recDesc) {
      if (isHealthy) {
        recTitle.textContent = "Triage Recommendation: No Acoustic Indicators Found";
        recDesc.textContent = "No motor speech disorder acoustic markers detected in this sample. Continue routine monitoring or consult for formal assessment if clinically indicated.";
      } else {
        recTitle.textContent = "Triage Recommendation: Clinical Follow-Up Advised";
        recDesc.textContent = "Acoustic biomarkers associated with dysarthric speech patterns were detected. Referral to a licensed Speech-Language Pathologist (SLP) for comprehensive evaluation is recommended.";
      }
    }

    // Signal Metadata
    if (signalMetricsRow && res.signal_metadata) {
      const m = res.signal_metadata;
      signalMetricsRow.innerHTML = `
        <div class="signal-stat"><span class="stat-label">Duration</span><span class="stat-val">${m.duration_sec}s</span></div>
        <div class="signal-stat"><span class="stat-label">Sample Rate</span><span class="stat-val">${m.sample_rate} Hz</span></div>
        <div class="signal-stat"><span class="stat-label">Peak Amp</span><span class="stat-val">${m.peak_amplitude}</span></div>
        <div class="signal-stat"><span class="stat-label">RMS Energy</span><span class="stat-val">${m.rms_energy}</span></div>
        <div class="signal-stat"><span class="stat-label">Clipping</span><span class="stat-val" style="color:${m.clipping_detected ? '#f43f5e' : '#34d399'}">${m.clipping_detected ? "Detected" : "None"}</span></div>
        <div class="signal-stat"><span class="stat-label">Channels</span><span class="stat-val">${m.channels || 1} (Mono)</span></div>
      `;
    }

    // Spectrogram rendering
    if (res.spectrogram && spectrogramCanvas) {
      drawSpectrogram(res.spectrogram);
    }

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* -----------------------------------------------------------------------
     9. SPECTROGRAM CANVAS PAINTER (Magma/Thermal heatmap)
  ----------------------------------------------------------------------- */
  function drawSpectrogram(spec) {
    const ctx = spectrogramCanvas.getContext("2d");
    const W = spectrogramCanvas.width;
    const H = spectrogramCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const values = spec.values;
    if (!values || values.length === 0) return;

    const nRows = values.length;
    const nCols = values[0].length;

    // Find min/max for normalization
    let vmin = Infinity, vmax = -Infinity;
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const v = values[r][c];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    const range = vmax - vmin || 1;

    const cellW = W / nCols;
    const cellH = H / nRows;

    // Magma-like colormap: dark blue → purple → red → orange → yellow
    function magmaColor(norm) {
      const t = Math.max(0, Math.min(1, norm));
      const r = Math.round(255 * Math.pow(t, 0.55));
      const g = Math.round(255 * Math.pow(t, 1.8));
      const b = Math.round(255 * (t < 0.5 ? t * 1.6 : Math.max(0, 1 - t * 2)));
      return `rgb(${r},${g},${b})`;
    }

    for (let row = 0; row < nRows; row++) {
      for (let col = 0; col < nCols; col++) {
        const norm = (values[row][col] - vmin) / range;
        const y = H - (row + 1) * cellH;
        const x = col * cellW;
        ctx.fillStyle = magmaColor(norm);
        ctx.fillRect(x, y, cellW + 0.8, cellH + 0.8);
      }
    }

    // Frequency axis guide lines
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 0.8;
    for (let i = 1; i <= 3; i++) {
      const y = H * (i / 4);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
  }

  /* -----------------------------------------------------------------------
     12. REPORT ACTIONS (Print & Copy JSON)
  ----------------------------------------------------------------------- */
  const printReportBtn = document.getElementById("printReportBtn");
  const copyJsonBtn    = document.getElementById("copyJsonBtn");

  if (printReportBtn) {
    printReportBtn.addEventListener("click", () => window.print());
  }

  if (copyJsonBtn) {
    copyJsonBtn.addEventListener("click", async () => {
      if (!lastResult) return;
      try {
        const jsonStr = JSON.stringify({
          timestamp: new Date().toISOString(),
          prediction: lastResult.prediction,
          score_healthy_pct: lastResult.score_healthy_pct,
          score_dysarthric_pct: lastResult.score_dysarthric_pct,
          model_score_pct: lastResult.model_score_pct,
          signal_metadata: lastResult.signal_metadata,
          model_used: lastResult.model_used
        }, null, 2);
        await navigator.clipboard.writeText(jsonStr);
        copyJsonBtn.textContent = "✓ Copied!";
        setTimeout(() => { copyJsonBtn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy Biomarker JSON`; }, 2200);
      } catch { /* clipboard unavailable */ }
    });
  }

  /* -----------------------------------------------------------------------
     12.5 INTERACTIVE SYMPTOM CHECKLIST
  ----------------------------------------------------------------------- */
  const symptomCheckboxes = document.querySelectorAll(".symptom-cb");
  const screenerRiskValue = document.getElementById("screenerRiskValue");
  const screenerRecText   = document.getElementById("screenerRecText");

  if (symptomCheckboxes.length > 0 && screenerRiskValue && screenerRecText) {
    function updateSymptomScore() {
      let count = 0;
      symptomCheckboxes.forEach(cb => { if (cb.checked) count++; });

      if (count === 0) {
        screenerRiskValue.textContent = "Low Indicator (0/5)";
        screenerRiskValue.style.color = "#34d399";
        screenerRecText.textContent = "No motor speech risk indicators reported. You can use the live microphone below to verify your acoustic stability.";
      } else if (count <= 2) {
        screenerRiskValue.textContent = `Mild Risk Indicator (${count}/5)`;
        screenerRiskValue.style.color = "#fbbf24";
        screenerRecText.textContent = `${count} mild speech motor symptom(s) selected. We recommend recording the Rainbow Passage below to check acoustic phonation stability.`;
      } else {
        screenerRiskValue.textContent = `High Clinical Risk (${count}/5)`;
        screenerRiskValue.style.color = "#f43f5e";
        screenerRecText.textContent = `${count} motor speech indicators reported. We recommend performing the live screening below and presenting the exported PDF biomarker report to a neurologist or SLP.`;
      }
    }

    symptomCheckboxes.forEach(cb => {
      cb.addEventListener("change", updateSymptomScore);
    });
  }

  /* -----------------------------------------------------------------------
     13. UTILITIES
  ----------------------------------------------------------------------- */
  function showError(msg) {
    if (errorMessage) errorMessage.textContent = msg;
    if (errorAlert)   errorAlert.style.display = "flex";
    if (resultsPlaceholder) resultsPlaceholder.style.display = "block";
    if (resultsContainer)   resultsContainer.style.display   = "none";
  }

  function hideError() {
    if (errorAlert) errorAlert.style.display = "none";
  }

});
