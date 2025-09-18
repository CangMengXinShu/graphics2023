#!/usr/bin/env python3
"""
R-CNN Architecture Diagram Generator - Chinese Academic Version
生成R-CNN模型架构的科研展示图（中文学术版本）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_rcnn_diagram_chinese():
    """Create R-CNN architecture diagram with Chinese text using PIL"""
    
    # Create a large canvas for high DPI
    width, height = 4800, 3600  # 300 DPI at 16x12 inches
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Define colors
    colors = {
        'stage1': '#E3F2FD',
        'stage2': '#F3E5F5',
        'stage3': '#E8F5E8',
        'border': '#424242',
        'text': '#212121',
        'red': '#FF0000',
        'lightgray': '#D3D3D3',
        'lightyellow': '#FFFFE0',
        'lightcoral': '#F08080',
        'lightgreen': '#90EE90',
        'white': '#FFFFFF'
    }
    
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def draw_rounded_rect(xy, size, fill_color, border_color, border_width=2):
        """Draw a rounded rectangle"""
        x1, y1 = xy
        w, h = size
        x2, y2 = x1 + w, y1 + h
        radius = 20
        
        # Draw main rectangle
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=hex_to_rgb(fill_color))
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=hex_to_rgb(fill_color))
        
        # Draw corners
        draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=hex_to_rgb(fill_color))
        draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=hex_to_rgb(fill_color))
        draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=hex_to_rgb(fill_color))
        draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=hex_to_rgb(fill_color))
        
        # Draw border
        if border_width > 0:
            # Top and bottom
            draw.rectangle([x1 + radius, y1, x2 - radius, y1 + border_width], fill=hex_to_rgb(border_color))
            draw.rectangle([x1 + radius, y2 - border_width, x2 - radius, y2], fill=hex_to_rgb(border_color))
            # Left and right
            draw.rectangle([x1, y1 + radius, x1 + border_width, y2 - radius], fill=hex_to_rgb(border_color))
            draw.rectangle([x2 - border_width, y1 + radius, x2, y2 - radius], fill=hex_to_rgb(border_color))
    
    # Scale factor for positioning
    scale = 300  # 300 pixels per "unit"
    
    # Draw input image
    draw_rounded_rect((150, 2400), (900, 750), colors['lightgray'], colors['border'])
    draw.text((600, 2700), 'Input Image', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=60)
    draw.text((600, 2800), '输入图像', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # Stage 1: Selective Search
    draw_rounded_rect((1500, 2400), (1200, 750), colors['stage1'], colors['border'])
    draw.text((2100, 2550), 'Stage 1: 选择性搜索', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=70)
    draw.text((2100, 2650), 'Selective Search', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=60)
    draw.text((2100, 2750), '生成候选区域', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=55)
    draw.text((2100, 2850), '~2000 region proposals', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # Draw candidate regions
    for x, y in [(3150, 2460), (3450, 2700), (3300, 2580)]:
        draw.rectangle([x, y, x+240, y+180], outline=hex_to_rgb(colors['red']), width=6)
    
    # Arrow: Input to Stage 1
    draw.polygon([(1440, 2775), (1350, 2745), (1350, 2805)], fill=hex_to_rgb(colors['border']))
    draw.rectangle([1050, 2760, 1440, 2790], fill=hex_to_rgb(colors['border']))
    
    # Stage 2: Feature Extraction
    draw_rounded_rect((1500, 1350), (1200, 750), colors['stage2'], colors['border'])
    draw.text((2100, 1500), 'Stage 2: 特征提取', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=70)
    draw.text((2100, 1600), 'Feature Extraction', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=60)
    draw.text((2100, 1700), '裁剪并缩放到固定尺寸', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=55)
    draw.text((2100, 1800), 'CNN (AlexNet): 227×227 → 4096-d', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # Feature matrix visualization
    draw_rounded_rect((3150, 1500), (450, 450), colors['lightyellow'], colors['border'])
    # Draw grid
    for i in range(8):
        for j in range(8):
            color_val = int(255 * np.random.random())
            grid_color = f'#{color_val//4:02x}{color_val//2:02x}{color_val:02x}'
            draw.rectangle([3170 + i*51, 1520 + j*51, 3170 + (i+1)*51, 1520 + (j+1)*51], 
                          fill=hex_to_rgb(grid_color), outline='white', width=2)
    
    draw.text((3375, 1380), '4096-d向量', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=45)
    
    # Arrow: Stage 1 to Stage 2
    draw.polygon([(2100, 2190), (2070, 2280), (2130, 2280)], fill=hex_to_rgb(colors['border']))
    draw.rectangle([2085, 2190, 2115, 2340], fill=hex_to_rgb(colors['border']))
    
    # Stage 3: Classification & Localization
    draw_rounded_rect((1500, 300), (1200, 750), colors['stage3'], colors['border'])
    draw.text((2100, 450), 'Stage 3: 分类与定位', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=70)
    draw.text((2100, 550), 'Classification & Localization', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=60)
    draw.text((2100, 650), 'SVM分类器 + 边界框回归', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=55)
    draw.text((2100, 750), 'Class scores & Bbox regression', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # SVM Classifier
    draw_rounded_rect((3150, 750), (540, 240), colors['lightcoral'], colors['border'])
    draw.text((3420, 870), 'SVM分类器', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # Bbox Regressor
    draw_rounded_rect((3150, 450), (540, 240), colors['lightgreen'], colors['border'])
    draw.text((3420, 570), '边界框回归', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # Arrow: Stage 2 to Stage 3
    draw.polygon([(2100, 1140), (2070, 1230), (2130, 1230)], fill=hex_to_rgb(colors['border']))
    draw.rectangle([2085, 1050, 2115, 1140], fill=hex_to_rgb(colors['border']))
    
    # Final Output
    draw_rounded_rect((4050, 1350), (660, 750), colors['lightyellow'], colors['border'])
    draw.text((4380, 1500), '最终输出', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=60)
    draw.text((4380, 1580), 'Final Output', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=55)
    draw.text((4380, 1680), '目标类别', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    draw.text((4380, 1730), '+', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    draw.text((4380, 1780), '边界框坐标', fill=hex_to_rgb(colors['text']), 
              anchor="mm", font_size=50)
    
    # Arrow: Stage 3 to Output
    draw.polygon([(3990, 1725), (3900, 1695), (3900, 1755)], fill=hex_to_rgb(colors['border']))
    draw.line([(2760, 675), (3990, 1725)], fill=hex_to_rgb(colors['border']), width=8)
    
    # Title
    draw.text((2400, 150), 'R-CNN (Region-based Convolutional Neural Networks) 架构图', 
              fill=hex_to_rgb(colors['text']), anchor="mm", font_size=90)
    draw.text((2400, 250), 'R-CNN Architecture for Object Detection', 
              fill=hex_to_rgb(colors['text']), anchor="mm", font_size=70)
    
    # Explanation box
    draw_rounded_rect((150, 750), (1050, 600), colors['white'], colors['border'])
    explanation = [
        "工作流程说明：",
        "1. 选择性搜索：从输入图像中",
        "   生成约2000个候选区域",
        "2. 特征提取：将每个候选区域",
        "   裁剪缩放至227×227，通过CNN",
        "   提取4096维特征向量",
        "3. 分类定位：使用SVM分类器进行",
        "   目标分类，同时使用边界框",
        "   回归器优化目标定位"
    ]
    
    for i, line in enumerate(explanation):
        draw.text((175, 780 + i*60), line, fill=hex_to_rgb(colors['text']), font_size=45)
    
    # Save image
    img.save('/home/runner/work/graphics2023/graphics2023/22351251余越/rcnn_architecture_diagram_chinese.png', 
             'PNG', dpi=(300, 300))
    
    print("Chinese R-CNN architecture diagram generated successfully!")
    print("File: rcnn_architecture_diagram_chinese.png")
    print("Specifications: 300 DPI, Chinese text version")

if __name__ == "__main__":
    create_rcnn_diagram_chinese()