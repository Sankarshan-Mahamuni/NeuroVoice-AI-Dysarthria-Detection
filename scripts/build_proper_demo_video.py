import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

def create_master_frame(raw_img_path, output_path, title, subtitle, caption):
    raw = Image.open(raw_img_path).convert('RGB')
    
    # Crop the 5px scrollbar off the right edge -> exact 1920 x 915
    w, h = raw.size
    clean_w = min(1920, w)
    clean_h = min(915, h)
    web_view = raw.crop((0, 0, clean_w, clean_h))
    if clean_w < 1920 or clean_h < 915:
        web_view = web_view.resize((1920, 915), Image.Resampling.LANCZOS)
    
    # Master frame: 1920 x 1080
    frame = Image.new('RGB', (1920, 1080), (15, 23, 42)) # dark slate
    
    # Paste web view in center (y: 75 to 990)
    frame.paste(web_view, (0, 75))
    
    draw = ImageDraw.Draw(frame)
    
    # Fonts
    font_bold = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 24)
    font_sub = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 16)
    font_pill = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 14)
    font_caption = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 18)
    
    # Top banner bar (0 to 75)
    draw.rectangle([0, 0, 1920, 75], fill=(15, 23, 42))
    draw.line([0, 74, 1920, 74], fill=(13, 148, 136), width=2) # teal border
    
    # Teal pill
    draw.rounded_rectangle([30, 20, 175, 54], radius=6, fill=(13, 148, 136))
    draw.text((45, 27), "NEUROVOICE", font=font_pill, fill=(255, 255, 255))
    
    # Title & Subtitle
    draw.text((195, 16), title, font=font_bold, fill=(255, 255, 255))
    draw.text((195, 46), subtitle, font=font_sub, fill=(148, 163, 184))
    
    # System badge on right
    draw.text((1520, 27), "87.36% Acc | 92.23% Spec", font=font_sub, fill=(52, 211, 153))
    
    # Bottom caption bar (990 to 1080)
    draw.rectangle([0, 990, 1920, 1080], fill=(15, 23, 42))
    draw.line([0, 990, 1920, 990], fill=(13, 148, 136), width=2)
    
    # Draw caption text centered
    draw.text((40, 1018), caption, font=font_caption, fill=(241, 245, 249))
    
    frame.save(output_path, quality=95)
    print(f"Generated master frame: {output_path}")

def main():
    temp_dir = 'C:/Users/Sankarshan/AppData/Local/Temp'
    reports_dir = 'E:/Dysarthia_Detection_Project/reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    scenes = [
        {
            "img": os.path.join(temp_dir, "real_01_hero.png"),
            "dur": 12.0,
            "title": "Clinical Dysarthria Screening Platform | System Overview",
            "subtitle": "Acoustic motor speech disorder detection using 144-dimensional biomarker extraction",
            "caption": "📌 Overview: Zero-leakage speaker-independent machine learning model achieving 87.36% test accuracy and 92.23% specificity."
        },
        {
            "img": os.path.join(temp_dir, "real_02_guide.png"),
            "dur": 11.0,
            "title": "Knowledge Center | Tab 1: Simplified Patient Guide & Methodology",
            "subtitle": "Etiology of dysarthria: how neurological injury impairs speech motor subsystems",
            "caption": "🧠 Methodology: Explaining how neuromuscular conditions impair the 4 speech subsystems: Respiration, Phonation, Articulation, and Prosody."
        },
        {
            "img": os.path.join(temp_dir, "real_03_subtypes.png"),
            "dur": 10.0,
            "title": "Knowledge Center | Tab 2: The 6 Clinical Dysarthria Subtypes",
            "subtitle": "Anatomical lesion sites, acoustic traits, and associated neurological pathologies",
            "caption": "📋 Clinical Taxonomy: Comprehensive mapping of Flaccid, Spastic, Ataxic, Hypokinetic, Hyperkinetic, and Mixed dysarthria."
        },
        {
            "img": os.path.join(temp_dir, "real_04_biomarkers.png"),
            "dur": 10.0,
            "title": "Knowledge Center | Tab 3: 144 Acoustic Biomarkers",
            "subtitle": "Physiologically grounded acoustic feature pipeline capturing vocal cord perturbation",
            "caption": "🔬 Feature Engineering: 40 MFCCs, 40 Mel Energies, Jitter, Shimmer, Pitch Perturbation, Spectral Flux, and Formant Dynamics."
        },
        {
            "img": os.path.join(temp_dir, "real_05_benchmarks.png"),
            "dur": 10.0,
            "title": "Knowledge Center | Tab 4: Multi-Model Benchmark Accuracies",
            "subtitle": "Comparative evaluation of 5 machine learning models evaluated on 13 unseen test speakers",
            "caption": "📊 Benchmark Results: Logistic Regression (87.36% Acc / 0.9545 AUC) outperforming Random Forest, Gradient Boosting, SVM, and MLP."
        },
        {
            "img": os.path.join(temp_dir, "real_06_pipeline.png"),
            "dur": 11.5,
            "title": "Knowledge Center | Tab 5: System Pipeline & Architecture",
            "subtitle": "End-to-end signal processing and classification workflow",
            "caption": "⚙️ System Pipeline: Audio Intake ➔ 16 kHz Mono Resample ➔ 144 Biomarkers ➔ Scaler Normalization ➔ Model Prediction ➔ Clinical Report."
        },
        {
            "img": os.path.join(temp_dir, "real_07_protocols.png"),
            "dur": 10.0,
            "title": "Knowledge Center | Tab 6: Clinical Protocols & Speech Drills",
            "subtitle": "Standardized speech elicitation tasks for objective clinical assessment",
            "caption": "🗣️ Diagnostic Protocols: Sustained Vowels (/a/), Diadochokinetic Rates (/pa-ta-ka/), Rainbow Passage Reading, and Monologue."
        },
        {
            "img": os.path.join(temp_dir, "real_08_screener.png"),
            "dur": 10.0,
            "title": "Clinical Triage | Interactive 5-Point Symptom Screener",
            "subtitle": "Point-of-care patient symptom checklist with real-time risk score calculation",
            "caption": "📝 Interactive Screener: Selected symptoms (slurred consonants, breathiness, tremor) dynamically update the live Risk Score Meter."
        },
        {
            "img": os.path.join(temp_dir, "real_09_testcase.png"),
            "dur": 14.2,
            "title": "Live Test Case | Dysarthric Phonation Classification Result",
            "subtitle": "Test Recording: 'dysarthric_speech_sample_moderate_02.wav' | Inference in 248ms",
            "caption": "🚨 Test Result: Model flags 'Pathological Markers Detected' (Rose-Red Warning) with 89.2% Dysarthric Risk score and SLP Referral."
        },
        {
            "img": os.path.join(temp_dir, "real_10_graph.png"),
            "dur": 18.0,
            "title": "Acoustic Graph | 40-Band Formant Mel Spectrogram Explained",
            "subtitle": "Thermal acoustic visualizer showing energy distribution and formant structure",
            "caption": "📈 Graph Explained: X-Axis = Time | Y-Axis = Frequency (0-8 kHz) | Color = Energy | Broken formants & noise confirm speech motor impairment."
        }
    ]
    
    concat_lines = []
    
    for i, sc in enumerate(scenes):
        out_name = f"master_scene_{i+1:02d}.png"
        out_path = os.path.join(reports_dir, out_name)
        create_master_frame(sc["img"], out_path, sc["title"], sc["subtitle"], sc["caption"])
        concat_lines.append(f"file '{out_name}'")
        concat_lines.append(f"duration {sc['dur']}")
    
    concat_lines.append(f"file '{os.path.basename(os.path.join(reports_dir, 'master_scene_10.png'))}'")
    
    concat_file_path = os.path.join(reports_dir, "master_scenes_concat.txt")
    with open(concat_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines) + "\n")
        
    print(f"Master frames and concat file created successfully at {concat_file_path}")

if __name__ == "__main__":
    main()
