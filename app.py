import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Configuration ──
st.set_page_config(
    page_title="Detecteur de dommages vehicules",
    page_icon="🚗",
    layout="wide"
)

MODEL_NAME = 'google/mobilenet_v2_1.0_224'
CLASS_NAMES = ['damaged', 'not_damaged']

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    model.eval()
    return model, device

def predict(model, device, img_tensor):
    inp = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=inp).logits
    exp_l = np.exp(logits.cpu().numpy()[0])
    probs = exp_l / exp_l.sum()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

# On crée une mini-classe adaptatrice pour Hugging Face
class HuggingFaceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits  # On extrait spécifiquement les logits

def generate_gradcam(model, device, img_tensor):
    # 1. On applique le wrapper au modèle original
    wrapped_model = HuggingFaceModelWrapper(model)
    
    # 2. On cible la couche (elle reste identique sur ton modèle chargé)
    target_layer = [model.mobilenet_v2.conv_stem]
    
    # 3. On passe le modèle enveloppé à GradCAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layer)
    
    inp = img_tensor.unsqueeze(0).to(device)
    grayscale = cam(input_tensor=inp, targets=None)
    return grayscale[0]

# ── Sidebar ──
st.sidebar.title("A propos")
st.sidebar.info("Pre-evaluation automatique par IA. L'avis d'un expert reste necessaire pour la validation du sinistre.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** MobileNet-V2")
st.sidebar.markdown("**Classes :** damaged / not_damaged")
st.sidebar.markdown("**Objectif :** AUC > 0.88")

# ── Page principale ──
st.title("🚗 Detecteur de dommages vehicules")
st.markdown("Uploadez une photo de vehicule pour obtenir une pre-evaluation automatique avec heatmap GradCAM.")

model, device = load_model()

uploaded_file = st.file_uploader(
    "Choisir une image du vehicule",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_tensor = val_transform(image_pil)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image_pil, use_container_width=True)

    with col2:
        st.subheader("Analyse IA")
        with st.spinner("Analyse en cours..."):
            pred_idx, probs = predict(model, device, img_tensor)
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

            # Affichage prediction
            color = "🔴" if pred_label == 'damaged' else "🟢"
            st.metric(label="Prediction", value=f"{color} {pred_label.upper()}",
                      delta=f"Confiance : {confidence:.1%}")

            st.progress(confidence)

            # GradCAM
            cam_map = generate_gradcam(model, device, img_tensor)
            mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img_rgb = (img_tensor * std_t + mean_t).permute(1,2,0).numpy()
            img_rgb = np.clip(img_rgb, 0, 1).astype(np.float32)
            overlay = show_cam_on_image(img_rgb, cam_map, use_rgb=True)

            st.subheader("Heatmap GradCAM")
            st.image(overlay, caption="Zones influencant la decision", use_container_width=True)

    st.markdown("---")
    st.error(
        "⚠️ DISCLAIMER : Cette pre-evaluation est un outil d'aide a la decision. "
        "Elle ne se substitue pas a l'expertise d'un professionnel qualifie. "
        "Toute decision d'indemnisation doit etre validee par un expert agree."
    )

    with st.expander("Comment ca marche ?"):
        st.markdown("""
        **MobileNet-V2** est un reseau de neurones convolutionnel entraine sur des milliers de photos de vehicules.
        
        - Il analyse la photo et produit un score de probabilite pour chaque classe.
        - **GradCAM** montre quelles zones de l'image ont influence la decision.
        - Les zones rouges/chaudes sont les plus importantes pour la prediction.
        - Si la heatmap pointe vers une zone irrelevante, ignorez la prediction.
        """)
else:
    st.info("Uploadez une image pour commencer l'analyse.")
