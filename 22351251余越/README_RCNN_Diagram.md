# R-CNN Architecture Diagram Generator

生成R-CNN模型架构的科研展示图，用于中文大论文介绍R-CNN前人工作。

## Generated Files

### 1. Main Diagram
- **File**: `rcnn_architecture_diagram.png`
- **Format**: PNG, 300 DPI
- **Size**: 16×12 inches (4800×3600 pixels)
- **Language**: English with clear technical annotations
- **Usage**: Suitable for international academic papers and presentations

### 2. Chinese Version
- **File**: `rcnn_architecture_diagram_chinese.png`  
- **Format**: PNG, 300 DPI
- **Size**: 16×12 inches (4800×3600 pixels)
- **Language**: Chinese with English technical terms
- **Usage**: Ideal for Chinese academic papers and domestic presentations

## Diagram Features

### Architecture Components Shown:
1. **Stage 1: Selective Search (选择性搜索)**
   - Input image processing
   - Generation of ~2000 region proposals
   - Visual representation of candidate regions with red bounding boxes

2. **Stage 2: Feature Extraction (特征提取)**
   - Crop and scale regions to 227×227 pixels
   - CNN (AlexNet) feature extraction
   - 4096-dimensional feature vector output
   - Visual matrix representation of extracted features

3. **Stage 3: Classification & Localization (分类与定位)**
   - SVM classifier for object classification
   - Bounding box regression for precise localization
   - Final output with class labels and coordinates

### Visual Design Elements:
- **Color Coding**: Different colors for each stage (blue, purple, green)
- **Professional Layout**: Clean, academic-style presentation
- **High Resolution**: 300 DPI for crisp printing in papers
- **Clear Arrows**: Show data flow between stages
- **Technical Details**: Include key parameters and specifications

## Usage Instructions

### For Academic Papers:
1. Use `rcnn_architecture_diagram.png` for international publications
2. Use `rcnn_architecture_diagram_chinese.png` for Chinese journals
3. Both images are print-ready at 300 DPI resolution

### Generation Scripts:
- `generate_rcnn_diagram.py`: Creates the main English version using matplotlib
- `generate_rcnn_diagram_chinese.py`: Creates Chinese version using PIL for better text handling

## Technical Specifications

- **Resolution**: 300 DPI (suitable for academic publishing)
- **Dimensions**: 16×12 inches
- **Color Space**: RGB
- **File Format**: PNG (lossless compression)
- **Font Rendering**: Optimized for academic readability

## Requirements

```bash
pip install matplotlib pillow numpy
```

## R-CNN Architecture Overview

The diagram illustrates the complete R-CNN (Region-based Convolutional Neural Networks) pipeline:

1. **Input**: Natural images of any size
2. **Region Proposals**: Selective search generates ~2000 potential object regions
3. **Feature Extraction**: Each region is processed through a CNN to extract features
4. **Classification**: SVM classifiers determine object categories
5. **Localization**: Bounding box regression refines object positions
6. **Output**: Final detections with class labels and precise bounding boxes

This visualization is perfect for explaining R-CNN methodology in academic contexts, showing both the high-level workflow and important technical details.