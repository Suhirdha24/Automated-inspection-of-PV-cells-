"""
☀️ Automated Inspection of Photovoltaic (PV) Cells using Deep Learning & Grad-CAM
Interactive Web Application for Real-Time Solar Cell Defect Detection & Explainability
"""

import os
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for web servers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio is required to run this app. Install it via: pip install gradio")

# ==============================================================================
# 1. Device Configuration & Model Loader
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        self.target_layer = target_layer
        
        self.f_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.b_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
        
    def _find_last_conv_layer(self, module):
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the model.")
        return last_conv

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min > 1e-8:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, probs

    def remove_hooks(self):
        self.f_hook.remove()
        self.b_hook.remove()


def load_inspection_model():
    """
    Loads trained checkpoint if available; falls back gracefully to pretrained backbone.
    """
    model = None
    checkpoint_desc = "Pretrained Backbone (Demo Mode)"
    
    # Priority 1: Checkpoints directory
    ckpt_paths = [
        os.path.join("checkpoints", "best_model.pth"),
        os.path.join("checkpoints", "best_model3.pth")
    ]
    
    active_ckpt = None
    for p in ckpt_paths:
        if os.path.exists(p):
            active_ckpt = p
            break
            
    if timm is not None:
        try:
            model = timm.create_model("legacy_xception", pretrained=(active_ckpt is None), num_classes=2)
            if active_ckpt:
                state_dict = torch.load(active_ckpt, map_location=DEVICE)
                model.load_state_dict(state_dict)
                checkpoint_desc = f"Trained Checkpoint: {os.path.basename(active_ckpt)}"
        except Exception:
            model = None

    if model is None:
        # Fallback to standard torchvision model
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if active_ckpt is None else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        if active_ckpt:
            try:
                state_dict = torch.load(active_ckpt, map_location=DEVICE)
                model.load_state_dict(state_dict)
                checkpoint_desc = f"Trained Checkpoint: {os.path.basename(active_ckpt)}"
            except Exception:
                pass
                
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint_desc

# Initialize model
INSPECTION_MODEL, MODEL_INFO = load_inspection_model()

# Image Preprocessing Transformation
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# 2. Image Processing & Metrics Helpers
# ==============================================================================
def enhance_contrast(img_np):
    """Adaptive luminance stretch to reveal hairline micro-cracks in EL cells."""
    lum = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    p_low, p_high = np.percentile(lum, (2, 98))
    if p_high > p_low:
        enhanced = np.clip((lum - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        enhanced = lum
    return np.stack([enhanced] * 3, axis=-1)

def compute_defect_metrics(cam, threshold=0.55):
    """Calculates defect area ratio and severity classification."""
    defect_mask = (cam >= threshold).astype(np.float32)
    defect_ratio = float(np.mean(defect_mask) * 100.0)
    peak_score = float(np.max(cam))
    
    if defect_ratio < 0.5:
        severity = "Nominal (No Significant Defect)"
        action = "✅ Cell meets quality standards. Cleared for module lamination."
        badge = "PASS"
        color = "#28a745"
    elif defect_ratio < 4.0:
        severity = "Mild Defect (Isolated Finger Interruption)"
        action = "⚠️ Minor flaw detected. Acceptable for low-grade module assembly or re-test."
        badge = "MONITOR"
        color = "#ffc107"
    elif defect_ratio < 12.0:
        severity = "Moderate Defect (Hairline Micro-Crack)"
        action = "❌ Crack detected. Risk of thermal hotspot failure. Re-route for manual inspection."
        badge = "REJECT"
        color = "#fd7e14"
    else:
        severity = "Severe Defect (Critical Fracture / Dead Area)"
        action = "⛔ Major inactive cell segment or severe breakage. Discard cell immediately."
        badge = "REJECT"
        color = "#dc3545"
        
    return defect_mask, defect_ratio, peak_score, severity, action, badge, color

# ==============================================================================
# 3. Main Inspection Function for Gradio
# ==============================================================================
def inspect_solar_cell(input_image, colormap="turbo", alpha=0.45, threshold=0.55):
    if input_image is None:
        return None, None, "⚠️ Please upload an image or select a sample solar cell from below.", {}

    # Convert to RGB PIL Image
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    rgb_image = input_image.convert("RGB")
    
    # Preprocess
    img_tensor = TRANSFORM(rgb_image).unsqueeze(0).to(DEVICE)
    
    # Resize original for display (224, 224)
    display_img = np.array(rgb_image.resize((224, 224)), dtype=np.float32) / 255.0

    # Run Grad-CAM
    grad_cam = GradCAM(INSPECTION_MODEL)
    cam, pred_class, probs = grad_cam.generate_cam(img_tensor)
    grad_cam.remove_hooks()

    class_names = ["Functional", "Defective"]
    is_defective = (pred_class == 1)
    
    # Metrics
    defect_mask, defect_ratio, peak_score, severity, action, badge, badge_color = compute_defect_metrics(cam, threshold)
    enhanced_img = enhance_contrast(display_img)
    
    # 1. Generate 5-Panel Quality Assurance Dashboard
    fig = plt.figure(figsize=(18, 4.0), dpi=130)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.28)

    # Panel 1: Raw EL
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(display_img)
    ax1.set_title("[1] Raw EL Cell Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Contrast Enhanced
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enhanced_img)
    ax2.set_title("[2] Enhanced Contrast\n(Micro-Crack Detail)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(cam, cmap=colormap, vmin=0.0, vmax=1.0)
    ax3.set_title(f"[3] Defect Heatmap\n(Grad-CAM {colormap})", fontsize=11, fontweight="bold")
    ax3.axis("off")
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # Panel 4: Segmented Contours & Overlay
    ax4 = fig.add_subplot(gs[0, 3])
    cmap_fn = plt.get_cmap(colormap)
    heatmap_rgb = cmap_fn(cam)[:, :, :3]
    blended = alpha * heatmap_rgb + (1.0 - alpha) * display_img
    ax4.imshow(blended)
    if np.any(defect_mask > 0):
        ax4.contour(defect_mask, levels=[0.5], colors=["#ff0055"], linewidths=[2.2])
    ax4.set_title("[4] Defect Localization\n(Segmented Contours)", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # Panel 5: Diagnostic Decision Bar
    ax5 = fig.add_subplot(gs[0, 4])
    bars = ax5.barh(["Functional", "Defective"], [probs[0], probs[1]], color=["#28a745", "#dc3545"], height=0.45)
    ax5.set_xlim(0.0, 1.15)
    ax5.set_xlabel("Confidence Score", fontsize=9, fontweight="bold")
    ax5.tick_params(axis="both", labelsize=9)
    for bar in bars:
        w = bar.get_width()
        ax5.text(w + 0.03, bar.get_y() + bar.get_height() / 2, f"{w:.1%}",
                 ha="left", va="center", fontsize=9, fontweight="bold")

    decision_bg = "#dc3545" if is_defective else "#28a745"
    decision_lbl = "REJECT (DEFECTIVE)" if is_defective else "PASS (FUNCTIONAL)"
    ax5.text(0.5, 1.35, f"QA DECISION: {decision_lbl}", transform=ax5.transAxes,
             fontsize=10, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=decision_bg, edgecolor="none"),
             ha="center", va="center")

    metrics_text = f"Defect Area: {defect_ratio:.1f}% | {severity}"
    ax5.text(0.5, -0.32, metrics_text, transform=ax5.transAxes,
             fontsize=9, fontweight="bold", color=badge_color,
             ha="center", va="center")

    plt.tight_layout()
    
    # Save figure to image buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    dashboard_img = Image.open(buf)
    plt.close(fig)

    # 2. Side-by-Side Blended Image
    overlay_pil = Image.fromarray((blended * 255.0).astype(np.uint8))

    # 3. Probabilities dictionary for Gradio Label
    conf_dict = {
        "Functional (No Defect)": float(probs[0]),
        "Defective (Micro-crack / Fault)": float(probs[1])
    }

    # 4. Markdown Diagnostic Report
    report_md = f"""
### 📋 Quality Assurance (QA) Inspection Report
* **Prediction:** <span style="color:{decision_bg}; font-weight:bold; font-size:16px;">{class_names[pred_class].upper()}</span> ({probs[pred_class]:.2%})
* **Decision Badge:** <span style="background-color:{decision_bg}; color:white; padding:4px 10px; border-radius:5px; font-weight:bold;">{badge}</span>
* **Defect Surface Area:** **`{defect_ratio:.2f}%`** of cell area affected
* **Peak Anomaly Intensity:** **`{peak_score:.3f}`** / 1.000
* **Defect Severity Triage:** <span style="color:{badge_color}; font-weight:bold;">{severity}</span>
* **Recommended Production Action:**  
  {action}

---
* **Inspection Engine:** `{MODEL_INFO}` on `{DEVICE.upper()}`
"""

    return dashboard_img, overlay_pil, report_md, conf_dict

# ==============================================================================
# 4. Build Gradio Web Interface
# ==============================================================================
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header-box {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(title="PV Solar Cell Defect Inspection", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column():
        gr.HTML("""
        <div class="header-box">
            <h1 style="margin: 0; font-size: 28px; color: #f8f9fa;">☀️ Automated Inspection of Photovoltaic (PV) Cells</h1>
            <p style="margin-top: 8px; font-size: 15px; color: #dcdde1;">
                Deep Learning Defect Classification & Explainable AI (Grad-CAM) Defect Localization
            </p>
        </div>
        """)
        
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="📸 Upload Solar Cell EL Image", height=280)
            
            with gr.Accordion("⚙️ Inspection Parameters", open=True):
                colormap_dropdown = gr.Dropdown(
                    choices=["turbo", "inferno", "jet", "magma", "plasma"],
                    value="turbo",
                    label="Grad-CAM Heatmap Colormap"
                )
                alpha_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                    label="Overlay Transparency (Alpha)"
                )
                threshold_slider = gr.Slider(
                    minimum=0.3, maximum=0.85, value=0.55, step=0.05,
                    label="Defect Segmentation Threshold (Tau)"
                )
                
            inspect_btn = gr.Button("🔍 Run Automated Quality Inspection", variant="primary", size="lg")
            
            # Examples section
            example_files = [
                ["sample_images/sample_functional_cell.png", "turbo", 0.45, 0.55],
                ["sample_images/sample_cracked_cell.png", "turbo", 0.45, 0.55],
                ["sample_images/sample_finger_interruption.png", "turbo", 0.45, 0.55]
            ]
            valid_examples = [ex for ex in example_files if os.path.exists(ex[0])]
            if valid_examples:
                gr.Examples(
                    examples=valid_examples,
                    inputs=[input_img, colormap_dropdown, alpha_slider, threshold_slider],
                    label="📁 Sample Test Cells (Click to Inspect)"
                )

        with gr.Column(scale=2):
            confidence_label = gr.Label(num_top_classes=2, label="📊 Classification Confidence")
            
            with gr.Tabs():
                with gr.TabItem("🏭 5-Panel Quality Dashboard"):
                    dashboard_output = gr.Image(label="Multi-View Inspection Dashboard", interactive=False)
                    
                with gr.TabItem("🔍 Defect Localization Overlay"):
                    overlay_output = gr.Image(label="Blended Heatmap Overlay & Defect Boundary", interactive=False)
                    
                with gr.TabItem("📋 Diagnostic Report"):
                    report_output = gr.Markdown()

    inspect_btn.click(
        fn=inspect_solar_cell,
        inputs=[input_img, colormap_dropdown, alpha_slider, threshold_slider],
        outputs=[dashboard_output, overlay_output, report_output, confidence_label]
    )

if __name__ == "__main__":
    print("Launching PV Cell Defect Inspection Web App...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
