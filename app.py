import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Détecteur de Dommages Véhicules",
    page_icon="🚗",
    layout="wide"
)

# ── Transform (identique à val_transform du notebook) ─────────
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Wrapper GradCAM (expose les logits en tenseur pur) ─────────
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

# ── Chargement du modèle (mis en cache — chargé une seule fois)
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(
        './model_weights',
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    # Réactiver les gradients pour GradCAM
    for param in model.parameters():
        param.requires_grad_(True)
    wrapped = ModelWrapper(model)
    wrapped.eval()
    return wrapped

def predict(wrapped_model, image_pil):
    """Retourne classe prédite, confiance, tenseur et toutes les probs."""
    tensor = VAL_TRANSFORM(image_pil).unsqueeze(0)
    with torch.no_grad():
        logits = wrapped_model(tensor)
    probs    = torch.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    classes  = wrapped_model.model.config.id2label
    return classes[pred_idx], probs[pred_idx].item(), tensor, probs

def get_gradcam(wrapped_model, tensor, image_pil):
    """Génère l'overlay GradCAM sur l'image."""
    target_layer = wrapped_model.model.mobilenet_v2.layer[-1].conv_3x3.convolution
    cam          = GradCAM(model=wrapped_model, target_layers=[target_layer])
    gray_cam     = cam(input_tensor=tensor)[0]
    img_rgb      = np.array(image_pil.resize((224, 224))) / 255.0
    return show_cam_on_image(img_rgb.astype(np.float32), gray_cam, use_rgb=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🚗 A propos")
st.sidebar.info(
    "Pré-évaluation automatique — "
    "l'avis d'un expert reste obligatoire pour la validation du sinistre."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modèle :** MobileNet-V2")
st.sidebar.markdown("**Classes :** `damaged` / `not_damaged`")
st.sidebar.markdown("**Dataset :** ~2 000 images smartphone")

with st.sidebar.expander("❓ Comment ça marche ?"):
    st.write(
        "Ce modèle utilise un réseau de neurones convolutionnel (MobileNet-V2) "
        "fine-tuné sur des photos de véhicules. Il analyse les patterns visuels "
        "(déformations, rayures, éclats) pour détecter un dommage.\n\n"
        "La **heatmap GradCAM** montre quelles zones de l'image ont le plus "
        "influencé la décision du modèle."
    )

# ── Page principale ───────────────────────────────────────────
st.title("🚗 Détecteur de Dommages Véhicules")
st.markdown(
    "Uploadez une photo de votre véhicule pour obtenir une **pré-évaluation automatique** "
    "des dommages visible en quelques secondes."
)

uploaded_file = st.file_uploader(
    "📸 Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG, PNG — photo prise depuis un smartphone"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Analyse")
        with st.spinner("Analyse en cours..."):
            try:
                wrapped_model = load_model()
                pred_class, confidence, tensor, probs = predict(wrapped_model, image)
                overlay = get_gradcam(wrapped_model, tensor, image)

                # ── Résultat ──
                is_damaged = pred_class == "damaged"
                icon  = "🔴" if is_damaged else "🟢"
                label = "ENDOMMAGÉ" if is_damaged else "INTACT"

                st.metric(
                    label="Prédiction",
                    value=f"{icon} {label}",
                    delta=f"Confiance : {confidence*100:.1f}%"
                )

                # Barre de confiance colorée
                conf_pct = int(confidence * 100)
                bar_color = "#e74c3c" if is_damaged else "#27ae60"
                st.markdown(
                    f"""
                    <div style="background:#eee;border-radius:8px;height:18px;width:100%;">
                      <div style="background:{bar_color};width:{conf_pct}%;height:18px;
                                  border-radius:8px;transition:width 0.4s;"></div>
                    </div>
                    <p style="font-size:12px;color:#666;margin-top:4px;">{conf_pct}% de confiance</p>
                    """,
                    unsafe_allow_html=True
                )

                # Probabilités détaillées
                classes = wrapped_model.model.config.id2label
                st.markdown("**Probabilités par classe :**")
                for idx, prob in enumerate(probs):
                    st.markdown(f"- `{classes[idx]}` : {prob.item()*100:.1f}%")

                # GradCAM
                st.markdown("**🗺️ Zones analysées (GradCAM) :**")
                st.image(overlay, use_container_width=True,
                         caption="Rouge = zones ayant le plus influencé la décision")

            except Exception as e:
                st.warning(
                    f"⚠️ Modèle non disponible. Sauvegardez d'abord le modèle "
                    f"avec `model.save_pretrained('./model_weights')`.\n\nErreur : {e}"
                )

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("---")
    st.error(
        "⚠️ **PRÉ-ÉVALUATION AUTOMATIQUE** — Cet outil est une aide à la décision uniquement. "
        "L'avis d'un expert terrain reste **obligatoire** pour valider tout sinistre. "
        "En cas de désaccord avec cette évaluation, utilisez le formulaire de contestation."
    )

else:
    # ── État initial : pas encore d'image ─────────────────────
    st.info("📸 Uploadez une photo de votre véhicule pour commencer l'analyse.")
    st.markdown(
        "**Conseils pour une bonne photo :**\n"
        "- Photographiez le véhicule de côté, en face ou en 3/4\n"
        "- Assurez-vous que la zone endommagée est bien visible\n"
        "- Évitez le contre-jour et les photos floues\n"
        "- Format recommandé : JPG ou PNG"
    )
