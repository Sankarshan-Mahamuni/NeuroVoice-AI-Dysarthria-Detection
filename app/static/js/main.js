/**
 * main.js
 * Frontend interactivity for NeuroVoice AI Dysarthria Detection.
 * Handles drag-and-drop, microphone recording, preset loading,
 * AJAX submission, and HTML5 Canvas spectrogram rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("audioFileInput");
  const recordBtn = document.getElementById("recordBtn");
  const recordBtnText = document.getElementById("recordBtnText");
  const analyzeForm = document.getElementById("analyzeForm");
  const submitBtn = document.getElementById("submitBtn");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const selectedSampleInput = document.getElementById("selectedSampleName");
  
  const filePreviewArea = document.getElementById("filePreviewArea");
  const dropZonePrompt = document.getElementById("dropZonePrompt");
  const previewFilename = document.getElementById("previewFilename");
  const previewFilesize = document.getElementById("previewFilesize");
  const audioPlayer = document.getElementById("audioPlayer");
  const clearFileBtn = document.getElementById("clearFileBtn");

  const processingStatus = document.getElementById("processingStatus");
  const statusStepText = document.getElementById("statusStepText");
  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");

  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const resultsContainer = document.getElementById("resultsContainer");
  const resultHeroCard = document.getElementById("resultHeroCard");
  const predictionTitle = document.getElementById("predictionTitle");
  const modelScoreVal = document.getElementById("modelScoreVal");
  const scoreHealthyText = document.getElementById("scoreHealthyText");
  const scoreDysarthricText = document.getElementById("scoreDysarthricText");
  const probFillHealthy = document.getElementById("probFillHealthy");
  const probFillDysarthric = document.getElementById("probFillDysarthric");
  const summaryNote = document.getElementById("summaryNote");
  const resultWarnings = document.getElementById("resultWarnings");
  const signalMetricsRow = document.getElementById("signalMetricsRow");
  const spectrogramCanvas = document.getElementById("spectrogramCanvas");

  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let recordTimer = null;
  let recordSeconds = 0;
  let recordedBlob = null;

  // --------------------------------------------------------------------------
  // Drag and Drop Handling
  // --------------------------------------------------------------------------
  if (dropZone) {
    ["dragenter", "dragover"].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    });
  }

  // File Input Change
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  // Clear Selection
  if (clearFileBtn) {
    clearFileBtn.addEventListener("click", () => {
      clearSelectedAudio();
    });
  }

  function handleFileSelect(file) {
    hideError();
    selectedSampleInput.value = "";
    recordedBlob = null;

    previewFilename.textContent = file.name;
    previewFilesize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB • ${file.type || 'audio'}`;

    const fileUrl = URL.createObjectURL(file);
    audioPlayer.src = fileUrl;

    dropZonePrompt.style.display = "none";
    filePreviewArea.style.display = "block";
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
    hideError();
  }

  // --------------------------------------------------------------------------
  // Preset Sample Loader
  // --------------------------------------------------------------------------
  window.loadPresetSample = function(filename, label) {
    hideError();
    fileInput.value = "";
    recordedBlob = null;
    selectedSampleInput.value = filename;

    previewFilename.textContent = `${label} (${filename.split('_').slice(-2).join('_')})`;
    previewFilesize.textContent = "Validation-speaker demonstration audio";

    audioPlayer.src = `/sample-audio/${encodeURIComponent(filename)}`;

    dropZonePrompt.style.display = "none";
    filePreviewArea.style.display = "block";
    submitBtn.disabled = false;

    // Automatically trigger analysis for seamless demo
    submitBtn.click();
  };

  document.querySelectorAll("[data-sample-name]").forEach((button) => {
    button.addEventListener("click", () => {
      window.loadPresetSample(button.dataset.sampleName, button.dataset.sampleLabel);
    });
  });

  // --------------------------------------------------------------------------
  // Microphone Live Recording (MediaRecorder API)
  // --------------------------------------------------------------------------
  if (recordBtn) {
    recordBtn.addEventListener("click", async () => {
      if (!isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    });
  }

  async function startRecording() {
    hideError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Microphone access is not supported by your current browser environment.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
        const audioUrl = URL.createObjectURL(recordedBlob);
        audioPlayer.src = audioUrl;

        previewFilename.textContent = `live_microphone_recording.webm`;
        previewFilesize.textContent = `${(recordedBlob.size / 1024).toFixed(1)} KB • Live Speech`;

        dropZonePrompt.style.display = "none";
        filePreviewArea.style.display = "block";
        submitBtn.disabled = false;
      };

      mediaRecorder.start();
      isRecording = true;
      recordBtn.classList.add("recording-active");
      recordSeconds = 0;
      recordBtnText.textContent = "Recording (0s)... Click to Stop";

      recordTimer = setInterval(() => {
        recordSeconds++;
        recordBtnText.textContent = `Recording (${recordSeconds}s)... Click to Stop`;
        if (recordSeconds >= 15) {
          stopRecording(); // Cap at 15s max
        }
      }, 1000);

    } catch (err) {
      showError("Could not access microphone: " + err.message);
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    isRecording = false;
    clearInterval(recordTimer);
    recordBtn.classList.remove("recording-active");
    recordBtnText.textContent = "Record Live Microphone";
  }

  // --------------------------------------------------------------------------
  // Form Submission & Inference AJAX
  // --------------------------------------------------------------------------
  if (analyzeForm) {
    analyzeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();

      const formData = new FormData(analyzeForm);

      // If a recorded microphone blob exists, attach it
      if (recordedBlob) {
        formData.set("audio_file", recordedBlob, "microphone_recording.webm");
      }

      // UI state: loading
      submitBtn.disabled = true;
      loadingSpinner.style.display = "inline-block";
      processingStatus.style.display = "block";
      statusStepText.textContent = "Ingesting and validating audio stream...";

      const stepTimer1 = setTimeout(() => {
        statusStepText.textContent = "Extracting the 144-dimensional acoustic feature vector...";
      }, 350);

      const stepTimer2 = setTimeout(() => {
        statusStepText.textContent = "Computing the model classification score...";
      }, 700);

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json"
          },
          body: formData
        });

        clearTimeout(stepTimer1);
        clearTimeout(stepTimer2);

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || "Analysis failed.");
        }

        renderResults(data);

      } catch (err) {
        showError(err.message);
      } finally {
        submitBtn.disabled = false;
        loadingSpinner.style.display = "none";
        processingStatus.style.display = "none";
      }
    });
  }

  // --------------------------------------------------------------------------
  // Result Rendering
  // --------------------------------------------------------------------------
  function renderResults(res) {
    resultsPlaceholder.style.display = "none";
    resultsContainer.style.display = "block";

    const isHealthy = res.predicted_target === 0;

    // Hero Card Styling
    resultHeroCard.className = `card result-hero-card ${isHealthy ? 'healthy' : 'dysarthric'}`;
    predictionTitle.textContent = res.prediction;
    predictionTitle.className = `prediction-title ${isHealthy ? 'healthy' : 'dysarthric'}`;

    modelScoreVal.textContent = `${res.model_score_pct}%`;
    scoreHealthyText.textContent = `${res.score_healthy_pct}%`;
    scoreDysarthricText.textContent = `${res.score_dysarthric_pct}%`;

    probFillHealthy.style.width = `${res.score_healthy_pct}%`;
    probFillDysarthric.style.width = `${res.score_dysarthric_pct}%`;

    summaryNote.textContent = res.summary_note;

    // Warnings
    if (res.warnings && res.warnings.length > 0) {
      resultWarnings.style.display = "block";
      resultWarnings.innerHTML = `<strong>Input Quality Advisory:</strong><ul>${res.warnings.map(w => `<li>${w}</li>`).join('')}</ul>`;
    } else {
      resultWarnings.style.display = "none";
    }

    // Signal Metadata Row
    const meta = res.signal_metadata;
    signalMetricsRow.innerHTML = `
      <div class="signal-stat"><span class="stat-label">Duration:</span><span class="stat-val">${meta.duration_sec}s</span></div>
      <div class="signal-stat"><span class="stat-label">Sample Rate:</span><span class="stat-val">${meta.sample_rate} Hz</span></div>
      <div class="signal-stat"><span class="stat-label">RMS Energy:</span><span class="stat-val">${meta.rms_energy}</span></div>
      <div class="signal-stat"><span class="stat-label">Clipping:</span><span class="stat-val">${meta.clipping_detected ? 'Yes' : 'None'}</span></div>
    `;

    // Render Canvas Spectrogram
    if (res.spectrogram && spectrogramCanvas) {
      drawSpectrogram(res.spectrogram);
    }

    // Scroll smoothly to results
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // --------------------------------------------------------------------------
  // Spectrogram Canvas Painter (Magma/Thermal heatmap)
  // --------------------------------------------------------------------------
  function drawSpectrogram(spec) {
    const ctx = spectrogramCanvas.getContext("2d");
    const width = spectrogramCanvas.width;
    const height = spectrogramCanvas.height;

    ctx.clearRect(0, 0, width, height);

    const values = spec.values;
    const nRows = values.length;
    const nCols = values[0].length;

    const cellW = width / nCols;
    const cellH = height / nRows;

    // Thermal color map (Black -> Purple -> Red -> Orange -> Yellow)
    function getColor(val) {
      const v = Math.max(0, Math.min(1, val));
      const r = Math.round(255 * Math.sqrt(v));
      const g = Math.round(255 * Math.pow(v, 2));
      const b = Math.round(255 * (1 - v) * v * 4);
      return `rgb(${r}, ${g}, ${b})`;
    }

    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        // Mel frequencies run bottom to top, so invert r
        const y = height - (r + 1) * cellH;
        const x = c * cellW;
        const val = values[r][c];

        ctx.fillStyle = getColor(val);
        ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
      }
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.style.display = "flex";
  }

  function hideError() {
    errorAlert.style.display = "none";
  }
});
