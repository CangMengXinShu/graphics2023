#!/usr/bin/env python3
"""
R-CNN Architecture Diagram Generator
Generate R-CNN model architecture diagram for academic paper introducing R-CNN prior work
生成R-CNN模型架构的科研展示图，用于中文大论文介绍R-CNN前人工作
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set font for better text rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_rcnn_diagram():
    """Create R-CNN architecture diagram"""
    
    # Create figure with high DPI for academic papers
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define color scheme
    colors = {
        'stage1': '#E3F2FD',  # Light blue - Selective Search
        'stage2': '#F3E5F5',  # Light purple - Feature Extraction  
        'stage3': '#E8F5E8',  # Light green - Classification & Localization
        'border': '#424242',  # Dark gray border
        'text': '#212121'     # Dark gray text
    }
    
    # Draw input image
    input_img = FancyBboxPatch(
        (0.5, 8), 3, 2.5,
        boxstyle="round,pad=0.1",
        facecolor='lightgray',
        edgecolor=colors['border'],
        linewidth=2
    )
    ax.add_patch(input_img)
    ax.text(2, 9.25, 'Input Image', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=colors['text'])
    
    # Stage 1: Selective Search
    stage1_box = FancyBboxPatch(
        (5, 8), 4, 2.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['stage1'],
        edgecolor=colors['border'],
        linewidth=2
    )
    ax.add_patch(stage1_box)
    ax.text(7, 9.8, 'Stage 1: Selective Search', ha='center', va='center',
            fontsize=14, fontweight='bold', color=colors['text'])
    ax.text(7, 9.3, 'Generate Region Proposals', ha='center', va='center',
            fontsize=12, color=colors['text'])
    ax.text(7, 8.7, '~2000 candidate regions', ha='center', va='center',
            fontsize=11, color=colors['text'])
    
    # Visualize candidate regions
    for i, (x, y) in enumerate([(10.5, 8.2), (11.5, 9.0), (11.0, 8.6)]):
        rect = patches.Rectangle((x, y), 0.8, 0.6, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # Arrow: Input to Stage 1
    ax.annotate('', xy=(4.8, 9.25), xytext=(3.7, 9.25),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['border']))
    
    # Stage 2: Feature Extraction
    stage2_box = FancyBboxPatch(
        (5, 4.5), 4, 2.5,
        boxstyle="round,pad=0.1", 
        facecolor=colors['stage2'],
        edgecolor=colors['border'],
        linewidth=2
    )
    ax.add_patch(stage2_box)
    ax.text(7, 6.3, 'Stage 2: Feature Extraction', ha='center', va='center',
            fontsize=14, fontweight='bold', color=colors['text'])
    ax.text(7, 5.8, 'Crop & Scale to Fixed Size', ha='center', va='center',
            fontsize=12, color=colors['text'])
    ax.text(7, 5.4, 'CNN (AlexNet): 227x227 -> 4096-d vector', ha='center', va='center',
            fontsize=11, color=colors['text'])
    
    # Feature vector visualization
    feature_matrix = FancyBboxPatch(
        (10.5, 5), 1.5, 1.5,
        boxstyle="round,pad=0.05",
        facecolor='lightyellow',
        edgecolor=colors['border'],
        linewidth=1
    )
    ax.add_patch(feature_matrix)
    
    # Draw feature vector matrix grid
    for i in range(8):
        for j in range(8):
            small_rect = patches.Rectangle((10.6 + i*0.17, 5.1 + j*0.17), 
                                         0.15, 0.15, 
                                         facecolor=plt.cm.Blues(np.random.random()),
                                         edgecolor='white', linewidth=0.5)
            ax.add_patch(small_rect)
    
    ax.text(11.25, 4.6, '4096-d vector', ha='center', va='center',
            fontsize=9, color=colors['text'])
    
    # Arrow: Stage 1 to Stage 2
    ax.annotate('', xy=(7, 7.3), xytext=(7, 7.8),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['border']))
    
    # Stage 3: Classification & Localization
    stage3_box = FancyBboxPatch(
        (5, 1), 4, 2.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['stage3'],
        edgecolor=colors['border'],
        linewidth=2
    )
    ax.add_patch(stage3_box)
    ax.text(7, 2.8, 'Stage 3: Classification & Localization', ha='center', va='center',
            fontsize=14, fontweight='bold', color=colors['text'])
    ax.text(7, 2.3, 'SVM Classifier + Bbox Regression', ha='center', va='center',
            fontsize=12, color=colors['text'])
    ax.text(7, 1.8, 'Class scores & Bounding box coordinates', ha='center', va='center',
            fontsize=11, color=colors['text'])
    
    # SVM Classifier
    svm_box = FancyBboxPatch(
        (10.5, 2.5), 1.8, 0.8,
        boxstyle="round,pad=0.05",
        facecolor='lightcoral',
        edgecolor=colors['border'],
        linewidth=1
    )
    ax.add_patch(svm_box)
    ax.text(11.4, 2.9, 'SVM Classifier', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['text'])
    
    # Bounding box regressor
    bbox_box = FancyBboxPatch(
        (10.5, 1.5), 1.8, 0.8,
        boxstyle="round,pad=0.05", 
        facecolor='lightgreen',
        edgecolor=colors['border'],
        linewidth=1
    )
    ax.add_patch(bbox_box)
    ax.text(11.4, 1.9, 'Bbox Regressor', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['text'])
    
    # Arrow: Stage 2 to Stage 3
    ax.annotate('', xy=(7, 3.7), xytext=(7, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['border']))
    
    # Final output
    output_box = FancyBboxPatch(
        (13.5, 4.5), 2.2, 2.5,
        boxstyle="round,pad=0.1",
        facecolor='lightyellow',
        edgecolor=colors['border'],
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(14.6, 6.3, 'Final Output', ha='center', va='center',
            fontsize=12, fontweight='bold', color=colors['text'])
    ax.text(14.6, 5.8, 'Object Class', ha='center', va='center',
            fontsize=11, color=colors['text'])
    ax.text(14.6, 5.5, '+', ha='center', va='center',
            fontsize=11, color=colors['text'])
    ax.text(14.6, 5.2, 'Bounding Box', ha='center', va='center',
            fontsize=11, color=colors['text'])
    ax.text(14.6, 4.9, 'Coordinates', ha='center', va='center',
            fontsize=11, color=colors['text'])
    
    # Arrow: Stage 3 to Output
    ax.annotate('', xy=(13.3, 5.75), xytext=(9.2, 2.25),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['border']))
    
    # Add title
    ax.text(8, 11.2, 'R-CNN (Region-based Convolutional Neural Networks) Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold', color=colors['text'])
    ax.text(8, 10.7, 'Three-Stage Object Detection Framework', 
            ha='center', va='center', fontsize=14, style='italic', color=colors['text'])
    
    # Add workflow explanation
    explanation_text = """Workflow Overview:
1. Selective Search: Generate approximately 2000 region proposals from input image
2. Feature Extraction: Crop and scale each region to 227x227, extract 4096-d features via CNN
3. Classification & Localization: Use SVM for object classification and bbox regressor for localization"""
    
    ax.text(0.5, 0.5, explanation_text, ha='left', va='bottom',
            fontsize=10, color=colors['text'], 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Add technical details in a separate box
    tech_details = FancyBboxPatch(
        (0.5, 2.5), 3.5, 2,
        boxstyle="round,pad=0.1",
        facecolor='#F5F5F5',
        edgecolor=colors['border'],
        linewidth=1
    )
    ax.add_patch(tech_details)
    ax.text(2.25, 3.8, 'Key Technical Details', ha='center', va='center',
            fontsize=11, fontweight='bold', color=colors['text'])
    ax.text(2.25, 3.4, '• AlexNet CNN backbone', ha='center', va='center',
            fontsize=9, color=colors['text'])
    ax.text(2.25, 3.1, '• Multi-class SVM classifier', ha='center', va='center',
            fontsize=9, color=colors['text'])
    ax.text(2.25, 2.8, '• Linear bbox regression', ha='center', va='center',
            fontsize=9, color=colors['text'])
    
    # Save image
    plt.tight_layout()
    plt.savefig('/home/runner/work/graphics2023/graphics2023/22351251余越/rcnn_architecture_diagram.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("R-CNN architecture diagram generated successfully!")
    print("File: rcnn_architecture_diagram.png")
    print("Specifications: 300 DPI, suitable for academic papers")

if __name__ == "__main__":
    create_rcnn_diagram()