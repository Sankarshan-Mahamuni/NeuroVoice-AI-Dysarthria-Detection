Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Rate = 0  # natural clear pace

$outputPath = "E:\Dysarthia_Detection_Project\reports\demo_narration_full_sync.wav"
$synth.SetOutputToWaveFile($outputPath)

$text = @"
Welcome to NeuroVoice AI, an open-source clinical screening platform for detecting motor speech disorders and dysarthria from voice recordings.

At the top of our dashboard, we display our verified system benchmarks: an 87.4 percent test accuracy, 92.2 percent specificity, and 0.95 ROC-AUC on held-out human speakers, following a strict zero-leakage evaluation protocol.

Let us explore our interactive Knowledge Center, starting with the Simplified Patient Guide. Here, patients and caregivers can understand how neurological injuries affect speech motor subsystems: respiration, phonation, articulation, and prosody.

Next, we review the six clinical dysarthria subtypes: Flaccid, Spastic, Ataxic, Hypokinetic, Hyperkinetic, and Mixed. Each subtype is mapped to its underlying anatomical lesion site, characteristic acoustic traits, and associated neurological conditions such as stroke, Parkinson's disease, or A.L.S.

In the Biomarkers tab, we detail the 144 acoustic features extracted by our signal processing engine. These include 40 Mel-frequency cepstral coefficients, log-Mel spectral energies, spectral flux, roll-off, zero-crossing rates, and pitch stability metrics that capture vocal cord perturbation.

The Model Benchmarks tab provides a comparative analysis of five machine learning architectures evaluated on identical speaker-independent splits. Our regularized Logistic Regression model achieved the highest generalization, outperforming Random Forests, Gradient Boosting, Support Vector Machines, and Multi-Layer Perceptrons.

The Diagnostic Flow tab illustrates our end-to-end processing pipeline, from audio ingestion, silence filtering, and 16-kilohertz resampling, through feature normalization, inference, and automated clinical triage.

In the Clinical Protocols tab, clinicians can access standardized speech elicitation drills, including sustained vowel phonation, diadochokinetic syllable alternating rates, and the standardized Rainbow Passage.

Below the Knowledge Center, we have implemented an Interactive 5-Point Symptom Screener. As a patient or clinician checks observed symptoms such as slurred consonants, vocal breathiness, or speech fatigue, the live risk meter updates dynamically to provide immediate clinical triage guidance.

Now, let us examine the Acoustic Diagnosis Engine. Clinicians can choose from verified demonstration recordings, upload custom audio files, or record live through the microphone.

When we select a healthy speech recording and trigger analysis, the model processes 144 biomarkers in under 300 milliseconds. The system confirms Normal Phonation with high confidence. The real-time Log-Mel Spectrogram displays smooth, stable formant bands across all frequencies, and the signal quality metrics confirm proper volume levels with zero clipping.

Next, let us analyze a dysarthric speech sample. The system immediately detects acoustic instability, changes the triage banner to Pathological Markers Detected, and provides an elevated risk score. The spectrogram visually reveals broken formant tracks and turbulent breath leakage characteristic of motor speech impairment.

Finally, clinicians can export a complete, printable clinical consultation report, or copy the raw biomarker JSON data for research and electronic health records.

All processing occurs completely locally on the server, ensuring full patient confidentiality. Thank you for watching this demonstration of NeuroVoice A.I.
"@

$synth.Speak($text)
$synth.Dispose()
Write-Output "Generated sync narration at $outputPath"
