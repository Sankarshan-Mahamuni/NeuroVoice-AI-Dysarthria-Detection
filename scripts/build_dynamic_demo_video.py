import os
import sys
from PIL import Image, ImageDraw, ImageFont

def make_scene_image(base_img_path, crop_box, output_path, title, subtitle, callout_lines=None):
    base_img = Image.open(base_img_path).convert('RGB')
    
    # Crop requested region
    cropped = base_img.crop(crop_box)
    cw, ch = cropped.size
    
    # Target frame size: 1920 x 1080
    target_w, target_h = 1920, 1080
    
    # Fit into 1920x1080
    scale = min(target_w / cw, target_h / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    frame = Image.new('RGB', (target_w, target_h), (15, 23, 42)) # dark slate background
    
    # Paste centered
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2 + 35 # leave room for top banner
    frame.paste(resized, (x_offset, y_offset))
    
    draw = ImageDraw.Draw(frame)
    
    # Load fonts
    font_bold = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 30)
    font_sub = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 20)
    font_tag = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 16)
    font_body = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    font_body_bold = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 19)
    
    # Top banner bar
    draw.rectangle([0, 0, target_w, 75], fill=(15, 23, 42))
    draw.line([0, 75, target_w, 75], fill=(13, 148, 136), width=3) # teal border
    
    # Badge pill on left
    draw.rounded_rectangle([30, 18, 190, 56], radius=6, fill=(13, 148, 136))
    draw.text((45, 26), "NEUROVOICE AI", font=font_tag, fill=(255, 255, 255))
    
    # Title & Subtitle
    draw.text((215, 14), title, font=font_bold, fill=(255, 255, 255))
    draw.text((215, 47), subtitle, font=font_sub, fill=(148, 163, 184))
    
    # Status on right
    draw.text((target_w - 380, 26), "Screening Platform Demo | 87.4% Accuracy", font=font_sub, fill=(52, 211, 153))
    
    # Callout card (if provided)
    if callout_lines:
        card_w = 640
        card_h = 42 + len(callout_lines) * 28 + 20
        card_x = target_w - card_w - 40
        card_y = target_h - card_h - 40
        
        # Semi-transparent dark glass card
        overlay = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=12, fill=(15, 23, 42, 240), outline=(13, 148, 136, 255), width=2)
        frame.paste(Image.alpha_composite(frame.convert('RGBA'), overlay).convert('RGB'))
        
        # Draw callout text
        draw = ImageDraw.Draw(frame)
        draw.text((card_x + 24, card_y + 16), callout_lines[0], font=font_body_bold, fill=(56, 189, 248))
        curr_y = card_y + 52
        for line in callout_lines[1:]:
            draw.text((card_x + 24, curr_y), line, font=font_body, fill=(241, 245, 249))
            curr_y += 28

    frame.save(output_path, quality=95)
    print(f"Generated scene image: {output_path}")

def main():
    temp_dir = 'C:/Users/Sankarshan/AppData/Local/Temp'
    reports_dir = 'E:/Dysarthia_Detection_Project/reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    scenes = [
        {
            "img": os.path.join(temp_dir, "shot_hero.png"),
            "crop": (0, 0, 1920, 850),
            "dur": 12.0,
            "title": "System Architecture & Validated Accuracies",
            "subtitle": "Zero Identity Leakage Protocol | 87.36% Accuracy | 92.23% Specificity on 13 Unseen Speakers",
            "callout": [
                "SYSTEM VALIDATION HIGHLIGHTS",
                "• Strict human-speaker-independent train/test split",
                "• Evaluated on 5,150 held-out clinical utterances",
                "• Model & preprocessing scaler versioned in repository"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_guide.png"),
            "crop": (320, 340, 1600, 1020),
            "dur": 12.0,
            "title": "Knowledge Center: Methodology & Patient Guide",
            "subtitle": "How Neurological Conditions Impair Speech Subsystems (Respiration, Phonation, Articulation, Prosody)",
            "callout": [
                "THE 4 MOTOR SUBSYSTEMS",
                "1. Respiration: Diminished breath capacity & stamina",
                "2. Phonation: Vocal cord tremor & pitch breaks",
                "3. Articulation: Imprecise consonant vowel transitions",
                "4. Prosody: Monotone inflection & unnatural rate"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_subtypes.png"),
            "crop": (320, 340, 1600, 1020),
            "dur": 10.0,
            "title": "Knowledge Center: The 6 Clinical Dysarthria Subtypes",
            "subtitle": "Anatomical Lesion Locations, Auditory Acoustic Traits, and Associated Pathologies",
            "callout": [
                "CLINICAL TAXONOMY MAPPING",
                "• Flaccid: Lower Motor Neuron lesion (Bulbar ALS)",
                "• Spastic: Bilateral UMN lesion (Multiple strokes)",
                "• Ataxic: Cerebellar lesion (Ataxia, Cadence breaks)",
                "• Hypokinetic: Basal Ganglia lesion (Parkinson's)"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_biomarkers.png"),
            "crop": (320, 340, 1600, 1020),
            "dur": 10.0,
            "title": "Knowledge Center: 144 Acoustic Biomarkers",
            "subtitle": "Physiologically Grounded Feature Pipeline Extracting Spectral, Harmonic, and Temporal Markers",
            "callout": [
                "FEATURE VECTOR ARCHITECTURE",
                "• 40 MFCC Mean & Standard Deviation Coefficients",
                "• 40 Log-Mel Spectral Energy Envelopes",
                "• Jitter, Shimmer, Pitch Perturbation & RMS Energy",
                "• Spectral Centroid, Spread, Flatness, Flux & Rolloff"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_benchmarks.png"),
            "crop": (320, 340, 1600, 1020),
            "dur": 10.0,
            "title": "Knowledge Center: Multi-Model Benchmark Accuracies",
            "subtitle": "Comparative Evaluation Across 5 Machine Learning Models on Identical Unseen Test Speakers",
            "callout": [
                "SPEAKER-INDEPENDENT BENCHMARK",
                "• Logistic Regression (L2, C=0.1): 87.36% Acc | 92.23% Spec",
                "• Random Forest: 84.12% Accuracy",
                "• Gradient Boosting: 85.20% Accuracy",
                "• Support Vector Machine: 86.40% Accuracy"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_pipeline.png"),
            "crop": (320, 340, 1600, 1020),
            "dur": 10.0,
            "title": "Knowledge Center: End-to-End Diagnostic Pipeline",
            "subtitle": "Raw Audio Ingestion -> 16 kHz Mono Resample -> Biomarker Extraction -> Standard Scaler -> Triage",
            "callout": [
                "REAL-TIME INGESTION PIPELINE",
                "• WebM / WAV / MP3 Ingestion & Silence Trimming",
                "• 16,000 Hz Resampling & Peak Normalization",
                "• Sub-300ms Clinical Inference & Report Output"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_screener.png"),
            "crop": (320, 480, 1600, 1180),
            "dur": 12.0,
            "title": "Clinical Triage: Interactive 5-Point Symptom Screener",
            "subtitle": "Interactive Checkpoints Updating Real-Time Risk Score Meter for Point-of-Care Assessment",
            "callout": [
                "5-POINT SYMPTOM CHECKLIST",
                "✔ Slurred or imprecise consonant articulation",
                "✔ Breathiness or hypophonic vocal volume",
                "✔ Vocal tremor or pitch instability",
                "➔ Dynamic Live Risk Meter: High Risk Tier"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_testcase_result.png"),
            "crop": (480, 1020, 1440, 1720),
            "dur": 14.0,
            "title": "Live Test Case Demonstration: Dysarthric Phonation",
            "subtitle": "Preset Audio Ingested: 'dysarthric_speech_sample_moderate_02.wav' | Instant Inference Result",
            "callout": [
                "INFERENCE RESULT & TRIAGE",
                "• Triage Pill: Pathological Markers Detected (Rose Red)",
                "• Prediction: Dysarthric Speech (89.2% Risk Score)",
                "• Clinical Recommendation: Immediate SLP Referral",
                "• Audio Latency: 248 ms on Standard CPU"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_testcase_result.png"),
            "crop": (860, 1600, 1440, 2320),
            "dur": 18.0,
            "title": "Acoustic Graph: Formant Mel Spectrogram Explained",
            "subtitle": "40-Band Thermal Mel Spectrogram Visualizing Energy Turbulence, Gaps, and Formant Breakdown",
            "callout": [
                "UNDERSTANDING THE GRAPH",
                "• X-Axis (Horizontal): Time (seconds)",
                "• Y-Axis (Vertical): Frequency Bins (0 to 8,000 Hz)",
                "• Color Heatmap: Yellow = Peak Energy | Dark = Silence",
                "• Finding: Broken formant transitions & breath turbulence",
                "  visually confirming motor speech impairment"
            ]
        },
        {
            "img": os.path.join(temp_dir, "shot_testcase_result.png"),
            "crop": (480, 1950, 1440, 2550),
            "dur": 5.0,
            "title": "Clinical Consultation Report & Open-Source Artifacts",
            "subtitle": "One-Click Printable PDF Summary, Raw Biomarker JSON Export, and MIT Open-Source Release",
            "callout": [
                "CONSULTATION & EXPORT",
                "• Complete structured medical report summary",
                "• 100% private, on-device local computation",
                "• Open-source on GitHub: Sankarshan-Mahamuni"
            ]
        }
    ]
    
    scene_files = []
    concat_lines = []
    
    for i, sc in enumerate(scenes):
        out_name = f"scene_{i+1:02d}.png"
        out_path = os.path.join(reports_dir, out_name)
        make_scene_image(
            sc["img"], sc["crop"], out_path,
            sc["title"], sc["subtitle"], sc.get("callout")
        )
        scene_files.append(out_path)
        concat_lines.append(f"file '{out_name}'")
        concat_lines.append(f"duration {sc['dur']}")
    
    # Concat file requires repeating the last file
    concat_lines.append(f"file '{os.path.basename(scene_files[-1])}'")
    
    concat_file_path = os.path.join(reports_dir, "scenes_concat.txt")
    with open(concat_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines) + "\n")
        
    print(f"Scenes prepared! Concat file written to {concat_file_path}")

if __name__ == "__main__":
    main()
