#!/usr/bin/env python3
"""
ECG AI Health Assistant - Single File Flask Application
A beautiful, medical-grade interface for ECG analysis
"""

from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import timm
from torchvision import transforms
import base64
from io import BytesIO
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variables
MODEL = None
LABEL_TO_ID = None
ID_TO_LABEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path=r'D:\projects\tests\ecg\trained_ecg_model.pth'):
    """Load the ECG classification model"""
    global MODEL, LABEL_TO_ID, ID_TO_LABEL
    
    if MODEL is not None:
        return MODEL
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=4)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        LABEL_TO_ID = checkpoint['label_to_id']
        ID_TO_LABEL = checkpoint['id_to_label']
        return MODEL
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def analyze_ecg_image(image):
    """Analyze ECG image and return prediction"""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = MODEL(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class_idx = output.argmax().item()
        confidence_score = probabilities[predicted_class_idx].item()
        predicted_label = ID_TO_LABEL[predicted_class_idx]
        all_probabilities = probabilities.cpu().numpy()
    
    return predicted_label, confidence_score, all_probabilities

def mark_ecg_image(img_array, prediction):
    """Mark ECG image based on prediction"""
    marked_img = img_array.copy()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    if prediction == 'Abnormal_Heartbeat':
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(marked_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    elif prediction == 'Myocardial_Infarction':
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=10, maxLineGap=15)
        if lines is not None:
            for line in lines[:12]:
                x1, y1, x2, y2 = line[0]
                cv2.line(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    elif prediction == 'Normal_ECG':
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
        if lines is not None:
            for line in lines[:10]:
                x1, y1, x2, y2 = line[0]
                cv2.line(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    elif prediction == 'Post_MI_History':
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 25 < area < 250:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(marked_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
    
    return marked_img

def get_explanation(prediction):
    """Get medical explanation for prediction"""
    explanations = {
        'Abnormal_Heartbeat': {
            'title': 'Abnormal Heartbeat Detected',
            'severity': 'warning',
            'icon': '⚠️',
            'findings': [
                'Irregular rhythm with inconsistent intervals',
                'Extra beats or premature contractions detected',
                'P, QRS, or T wave disruptions observed',
                'Non-standard electrical conduction patterns'
            ],
            'recommendation': 'Consult with a cardiologist for further evaluation and possible rhythm monitoring.'
        },
        'Myocardial_Infarction': {
            'title': 'Myocardial Infarction Signs',
            'severity': 'critical',
            'icon': '🚨',
            'findings': [
                'ST-segment elevation indicating heart muscle damage',
                'Abnormal Q-waves suggesting tissue necrosis',
                'T-wave inversions indicating ischemia',
                'Altered ST and T segments detected'
            ],
            'recommendation': 'IMMEDIATE medical attention required. Contact emergency services or go to the nearest ER.'
        },
        'Normal_ECG': {
            'title': 'Normal ECG Pattern',
            'severity': 'normal',
            'icon': '✓',
            'findings': [
                'Regular rhythm with consistent intervals',
                'Normal P-QRS-T wave sequence',
                'Appropriate wave amplitudes',
                'No significant abnormalities detected'
            ],
            'recommendation': 'Continue regular health check-ups and maintain a heart-healthy lifestyle.'
        },
        'Post_MI_History': {
            'title': 'Post-MI Changes Detected',
            'severity': 'monitor',
            'icon': '📋',
            'findings': [
                'Evidence of previous cardiac event',
                'Persistent ECG abnormalities from scar tissue',
                'Altered electrical conduction pathways',
                'Permanent changes from prior myocardial damage'
            ],
            'recommendation': 'Regular follow-up with cardiologist. Continue prescribed medications and lifestyle modifications.'
        }
    }
    return explanations.get(prediction, explanations['Normal_ECG'])

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG AI Health Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0066FF;
            --primary-dark: #0052CC;
            --secondary: #00D9FF;
            --success: #00C853;
            --warning: #FF9800;
            --critical: #FF3D00;
            --bg-primary: #0A0E1A;
            --bg-secondary: #111827;
            --bg-tertiary: #1F2937;
            --text-primary: #F9FAFB;
            --text-secondary: #D1D5DB;
            --text-muted: #9CA3AF;
            --border: rgba(255, 255, 255, 0.08);
            --shadow: rgba(0, 0, 0, 0.5);
            --glow: rgba(0, 102, 255, 0.4);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated Background */
        .bg-grid {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 102, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 102, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: 0;
            animation: gridMove 20s linear infinite;
        }

        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }

        .bg-gradient {
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
        }

        .gradient-1 {
            top: -200px;
            left: -200px;
            background: var(--primary);
            animation: float 15s ease-in-out infinite;
        }

        .gradient-2 {
            bottom: -200px;
            right: -200px;
            background: var(--secondary);
            animation: float 12s ease-in-out infinite reverse;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(100px, 100px) scale(1.2); }
        }

        /* Header */
        header {
            position: relative;
            z-index: 10;
            padding: 2rem 3rem;
            background: rgba(17, 24, 39, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 4px 20px var(--glow);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 4px 20px var(--glow); }
            50% { transform: scale(1.05); box-shadow: 0 6px 30px var(--glow); }
        }

        .logo-text h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .logo-text p {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .status-badge {
            padding: 0.5rem 1rem;
            background: rgba(0, 200, 83, 0.1);
            border: 1px solid rgba(0, 200, 83, 0.3);
            border-radius: 20px;
            color: var(--success);
            font-size: 0.875rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: blink 2s ease-in-out infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* Main Container */
        .container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 3rem;
        }

        /* Upload Section */
        .upload-section {
            background: var(--bg-secondary);
            border: 2px dashed var(--border);
            border-radius: 24px;
            padding: 4rem;
            text-align: center;
            transition: all 0.3s ease;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .upload-section:hover {
            border-color: var(--primary);
            background: rgba(0, 102, 255, 0.02);
        }

        .upload-section.dragover {
            border-color: var(--secondary);
            background: rgba(0, 217, 255, 0.05);
            transform: scale(1.02);
        }

        .upload-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            box-shadow: 0 8px 30px var(--glow);
        }

        .upload-section h2 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .upload-section p {
            color: var(--text-secondary);
            margin-bottom: 2rem;
            font-size: 1rem;
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
        }

        input[type="file"] {
            display: none;
        }

        .upload-btn {
            padding: 1rem 2.5rem;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 102, 255, 0.3);
        }

        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 102, 255, 0.5);
        }

        /* Progress Steps */
        .progress-steps {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin-bottom: 3rem;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.6s ease forwards 0.2s;
        }

        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .step {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .step.active {
            border-color: var(--primary);
            background: rgba(0, 102, 255, 0.05);
        }

        .step.complete {
            border-color: var(--success);
            background: rgba(0, 200, 83, 0.05);
        }

        .step-number {
            width: 40px;
            height: 40px;
            margin: 0 auto 0.75rem;
            background: var(--bg-tertiary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.125rem;
            transition: all 0.3s ease;
        }

        .step.active .step-number {
            background: var(--primary);
            box-shadow: 0 4px 20px var(--glow);
        }

        .step.complete .step-number {
            background: var(--success);
        }

        .step-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .step.active .step-title,
        .step.complete .step-title {
            color: var(--text-primary);
        }

        /* Results Section */
        .results-section {
            display: none;
            animation: fadeIn 0.6s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .image-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            overflow: hidden;
        }

        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }

        .panel-title {
            font-size: 1.125rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .image-container {
            width: 100%;
            aspect-ratio: 4/3;
            background: var(--bg-primary);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }

        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Analysis Panel */
        .analysis-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
        }

        .diagnosis-card {
            background: var(--bg-tertiary);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border-left: 4px solid var(--primary);
        }

        .diagnosis-card.critical {
            border-left-color: var(--critical);
            background: rgba(255, 61, 0, 0.05);
        }

        .diagnosis-card.warning {
            border-left-color: var(--warning);
            background: rgba(255, 152, 0, 0.05);
        }

        .diagnosis-card.normal {
            border-left-color: var(--success);
            background: rgba(0, 200, 83, 0.05);
        }

        .diagnosis-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .diagnosis-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            background: rgba(255, 255, 255, 0.05);
        }

        .diagnosis-title {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .confidence-score {
            margin: 1.5rem 0;
        }

        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .confidence-bar {
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 4px;
            transition: width 1s ease;
        }

        .findings {
            margin: 2rem 0;
        }

        .findings-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }

        .finding-item {
            padding: 0.75rem;
            background: var(--bg-primary);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }

        .finding-bullet {
            width: 6px;
            height: 6px;
            background: var(--primary);
            border-radius: 50%;
            margin-top: 0.5rem;
            flex-shrink: 0;
        }

        .recommendation {
            background: rgba(0, 102, 255, 0.1);
            border: 1px solid rgba(0, 102, 255, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }

        .recommendation-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--secondary);
        }

        /* Probabilities */
        .probabilities {
            margin-top: 2rem;
        }

        .prob-item {
            margin-bottom: 1rem;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }

        .prob-label {
            font-weight: 500;
        }

        .prob-value {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }

        .prob-bar {
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 3px;
            transition: width 0.8s ease;
        }

        /* Loading Spinner */
        .loading {
            display: none;
            text-align: center;
            padding: 3rem;
        }

        .spinner {
            width: 60px;
            height: 60px;
            margin: 0 auto 1rem;
            border: 4px solid var(--bg-tertiary);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 968px) {
            .results-grid {
                grid-template-columns: 1fr;
            }

            .progress-steps {
                grid-template-columns: 1fr;
            }

            header {
                padding: 1.5rem;
            }

            .container {
                padding: 1.5rem;
            }

            .upload-section {
                padding: 2rem;
            }
        }

        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <!-- Background Elements -->
    <div class="bg-grid"></div>
    <div class="bg-gradient gradient-1"></div>
    <div class="bg-gradient gradient-2"></div>

    <!-- Header -->
    <header>
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">❤️</div>
                <div class="logo-text">
                    <h1>ECG AI Health Assistant</h1>
                    <p>AI-Powered Cardiac Analysis</p>
                </div>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span id="model-status">AI Model Ready</span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <div class="container">
        <!-- Progress Steps -->
        <div class="progress-steps">
            <div class="step" id="step-1">
                <div class="step-number">1</div>
                <div class="step-title">Upload ECG</div>
            </div>
            <div class="step" id="step-2">
                <div class="step-number">2</div>
                <div class="step-title">Validation</div>
            </div>
            <div class="step" id="step-3">
                <div class="step-number">3</div>
                <div class="step-title">AI Analysis</div>
            </div>
            <div class="step" id="step-4">
                <div class="step-number">4</div>
                <div class="step-title">Prediction</div>
            </div>
            <div class="step" id="step-5">
                <div class="step-number">5</div>
                <div class="step-title">Results</div>
            </div>
        </div>

        <!-- Upload Section -->
        <div class="upload-section" id="upload-section">
            <div class="upload-icon">📊</div>
            <h2>Upload ECG Image for Analysis</h2>
            <p>Drag and drop your ECG image here or click to browse</p>
            <div class="file-input-wrapper">
                <input type="file" id="ecg-file" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('ecg-file').click()">
                    Select ECG Image
                </button>
            </div>
        </div>

        <!-- Loading Section -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>Analyzing ECG...</h3>
            <p style="color: var(--text-secondary); margin-top: 0.5rem;">Processing image and detecting patterns</p>
        </div>

        <!-- Results Section -->
        <div class="results-section" id="results">
            <div class="results-grid">
                <!-- Original Image -->
                <div class="image-panel">
                    <div class="panel-header">
                        <div class="panel-title">
                            <span>📄</span>
                            Original ECG
                        </div>
                    </div>
                    <div class="image-container">
                        <img id="original-image" src="" alt="Original ECG">
                    </div>
                </div>

                <!-- Marked Image -->
                <div class="image-panel">
                    <div class="panel-header">
                        <div class="panel-title">
                            <span>🔍</span>
                            AI-Annotated ECG
                        </div>
                    </div>
                    <div class="image-container">
                        <img id="marked-image" src="" alt="Marked ECG">
                    </div>
                </div>
            </div>

            <!-- Analysis Panel -->
            <div class="analysis-panel">
                <div class="diagnosis-card" id="diagnosis-card">
                    <div class="diagnosis-header">
                        <div class="diagnosis-icon" id="diagnosis-icon">❤️</div>
                        <div>
                            <div class="diagnosis-title" id="diagnosis-title">Analysis Results</div>
                        </div>
                    </div>

                    <div class="confidence-score">
                        <div class="confidence-label">
                            <span>Confidence Score</span>
                            <span id="confidence-value">0%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="findings">
                        <div class="findings-title">Key Findings</div>
                        <div id="findings-list"></div>
                    </div>

                    <div class="recommendation">
                        <div class="recommendation-title">📋 Medical Recommendation</div>
                        <div id="recommendation-text"></div>
                    </div>
                </div>

                <div class="probabilities">
                    <h3 style="margin-bottom: 1rem; font-size: 1.125rem;">Probability Distribution</h3>
                    <div id="probabilities-list"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadSection = document.getElementById('upload-section');
        const fileInput = document.getElementById('ecg-file');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const steps = document.querySelectorAll('.step');

        // Drag and drop functionality
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFileUpload(file);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleFileUpload(file);
            }
        });

        function updateStep(stepNum) {
            steps.forEach((step, index) => {
                step.classList.remove('active', 'complete');
                if (index < stepNum - 1) {
                    step.classList.add('complete');
                } else if (index === stepNum - 1) {
                    step.classList.add('active');
                }
            });
        }

        async function handleFileUpload(file) {
            // Update UI
            updateStep(1);
            uploadSection.style.display = 'none';
            loading.style.display = 'block';
            results.style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
                updateStep(2);
                await new Promise(resolve => setTimeout(resolve, 500));
                
                updateStep(3);
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                updateStep(4);
                const data = await response.json();

                if (data.success) {
                    updateStep(5);
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                    resetUI();
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during analysis');
                resetUI();
            }
        }

        function displayResults(data) {
            loading.style.display = 'none';
            results.style.display = 'block';

            // Display images
            document.getElementById('original-image').src = 'data:image/jpeg;base64,' + data.original_image;
            document.getElementById('marked-image').src = 'data:image/jpeg;base64,' + data.marked_image;

            // Set diagnosis card styling
            const diagCard = document.getElementById('diagnosis-card');
            diagCard.className = 'diagnosis-card ' + data.explanation.severity;

            document.getElementById('diagnosis-icon').textContent = data.explanation.icon;
            document.getElementById('diagnosis-title').textContent = data.explanation.title;

            // Confidence score
            const confidence = Math.round(data.confidence * 100);
            document.getElementById('confidence-value').textContent = confidence + '%';
            document.getElementById('confidence-fill').style.width = confidence + '%';

            // Findings
            const findingsList = document.getElementById('findings-list');
            findingsList.innerHTML = '';
            data.explanation.findings.forEach(finding => {
                const item = document.createElement('div');
                item.className = 'finding-item';
                item.innerHTML = `
                    <div class="finding-bullet"></div>
                    <div>${finding}</div>
                `;
                findingsList.appendChild(item);
            });

            // Recommendation
            document.getElementById('recommendation-text').textContent = data.explanation.recommendation;

            // Probabilities
            const probsList = document.getElementById('probabilities-list');
            probsList.innerHTML = '';
            const labels = ['Abnormal_Heartbeat', 'Myocardial_Infarction', 'Normal_ECG', 'Post_MI_History'];
            labels.forEach((label, index) => {
                const prob = data.probabilities[index];
                const percentage = Math.round(prob * 100);
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <div class="prob-header">
                        <span class="prob-label">${label.replace(/_/g, ' ')}</span>
                        <span class="prob-value">${percentage}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${percentage}%"></div>
                    </div>
                `;
                probsList.appendChild(item);
            });
        }

        function resetUI() {
            uploadSection.style.display = 'block';
            loading.style.display = 'none';
            results.style.display = 'none';
            updateStep(1);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if model is loaded
        if MODEL is None:
            return jsonify({'success': False, 'error': 'Model not loaded. Please ensure trained_ecg_model.pth is available.'})
        
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        
        # Read image
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        img_array = np.array(image.convert('RGB'))
        
        # Analyze image
        prediction, confidence, probabilities = analyze_ecg_image(image)
        
        # Mark image
        marked_array = mark_ecg_image(img_array, prediction)
        
        # Convert images to base64
        original_base64 = image_to_base64(img_array)
        marked_base64 = image_to_base64(marked_array)
        
        # Get explanation
        explanation = get_explanation(prediction)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'original_image': original_base64,
            'marked_image': marked_base64,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    # Load model on startup
    print("Loading ECG AI model...")
    load_model('trained_ecg_model.pth')
    print("Model loaded successfully!")
    
    # Run Flask app
    print("\n" + "="*60)
    print("🏥 ECG AI Health Assistant Starting...")
    print("="*60)
    print("📍 Access the application at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)