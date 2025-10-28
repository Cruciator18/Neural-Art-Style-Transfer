import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import copy

# --- Streamlit Configuration ---
st.set_page_config(layout="wide")
st.title(" Neural Style Transfer")

st.sidebar.header("Configuration")
IMAGE_SIZE = st.sidebar.slider("Image Size", 128, 512, 256, step=64)
ALPHA = st.sidebar.slider("Content Weight (Alpha)", 0.1, 10.0, 1.0, step=0.1)
BETA = st.sidebar.select_slider(
    "Style Weight (Beta)",
    options=[1e4, 1e5, 1e6, 1e7, 1e8],
    value=1e6
)
NUM_STEPS = st.sidebar.slider("Optimization Steps", 100, 1000, 400, step=50) # Increased max steps
LEARNING_RATE = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.001, 0.005, 0.01, 0.015, 0.02, 0.05], # Added 0.015
    value=0.015
)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: **{device}**")
if str(device) == "cuda":
    st.sidebar.write(f"GPU: **{torch.cuda.get_device_name(0)}**")
else:
    st.sidebar.warning("CUDA (GPU) not available, using CPU. This will be very slow!")

# --- Image Loading & Preprocessing ---

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# Corrected typo in std deviation from user script
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device) 

def get_loader(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def load_image_st(uploaded_file, image_size):
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            loader = get_loader(image_size)
            image = loader(image).unsqueeze(0)
            return image.to(device, torch.float)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None
    return None

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    mean_cpu = imagenet_mean.cpu()
    std_cpu = imagenet_std.cpu()
    image = image * std_cpu.view(3, 1, 1) + mean_cpu.view(3, 1, 1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    return image

# --- VGG Model & Feature Extraction ---

@st.cache_resource
def get_vgg_model():
    st.write("Cache miss: Loading VGG19 model...")
    vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT) # Correct way to load weights
    vgg19_features = vgg19.features.to(device).eval()
    for param in vgg19_features.parameters():
        param.requires_grad_(False)
    st.write("VGG19 model loaded and frozen.")
    return vgg19_features

class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg19_features):
        super().__init__()
        self.content_layers_default = ['21']
        self.style_layers_default = ['0', '5', '10', '19', '28']
        all_layers_indices = [int(l) for l in self.content_layers_default + self.style_layers_default]
        max_layer_idx = max(all_layers_indices)
        self.features = nn.Sequential(*[vgg19_features[i] for i in range(max_layer_idx + 1)])

    def forward(self, x):
        content_features_out = {}
        style_features_out = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.content_layers_default:
                content_features_out['content'] = x
            if name in self.style_layers_default:
                style_features_out[name] = x
        return content_features_out, style_features_out

# --- Style Loss (Gram Matrix) ---
def gram_matrix(input_tensor):
    B, C, H, W = input_tensor.size()
    features = input_tensor.view(C, H * W)
    G = torch.mm(features, features.t())
    return G.div(C * H * W)

# --- Main Style Transfer Function ---
def run_style_transfer(content_img_tensor, style_img_tensor, model_extractor, num_steps, alpha, beta, lr):
    with torch.no_grad():
        target_content_features, _ = model_extractor(content_img_tensor)
        target_content_tensor = target_content_features['content']
        _, target_style_features = model_extractor(style_img_tensor)
        target_style_grams = {name: gram_matrix(feat) for name, feat in target_style_features.items()}

    target_image = copy.deepcopy(content_img_tensor).requires_grad_(True)
    optimizer = optim.Adam([target_image], lr=lr)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        current_content_features, current_style_features = model_extractor(target_image)

        content_loss = F.mse_loss(current_content_features['content'], target_content_tensor)

        style_loss = 0
        for layer_name in current_style_features:
            current_gram = gram_matrix(current_style_features[layer_name])
            target_gram = target_style_grams[layer_name]
            style_loss += F.mse_loss(current_gram, target_gram)

        total_loss = alpha * content_loss + beta * style_loss
        total_loss.backward()
        optimizer.step()

        progress_percentage = int((step / num_steps) * 100)
        progress_bar.progress(progress_percentage)
        status_text.text(f"Step {step}/{num_steps} | Total Loss: {total_loss.item():.2f}")

    status_text.text("Optimization finished!")
    progress_bar.empty()
    return target_image.detach()

# --- Streamlit UI Layout ---

col1, col2 = st.columns(2)

with col1:
    st.header("Content Image")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    if content_file:
        content_pil = Image.open(content_file)
        st.image(content_pil, caption="Uploaded Content Image", use_container_width=True)

with col2:
    st.header("Style Image")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
    if style_file:
        style_pil = Image.open(style_file)
        st.image(style_pil, caption="Uploaded Style Image", use_container_width=True)

st.markdown("---")

stylize_button = st.button(" Stylize Image", type="primary", disabled=(not content_file or not style_file))

if stylize_button and content_file and style_file:
    content_img = load_image_st(content_file, IMAGE_SIZE)
    style_img = load_image_st(style_file, IMAGE_SIZE)

    if content_img is not None and style_img is not None:
        vgg_base_features = get_vgg_model()
        model = VGGFeatureExtractor(vgg_base_features)

        st.header("Output Image")
        with st.spinner('🎨 Applying artistic style... Please wait.'):
            output_tensor = run_style_transfer(
                content_img, style_img, model,
                num_steps=NUM_STEPS, alpha=ALPHA, beta=BETA, lr=LEARNING_RATE
            )
            output_image_pil = tensor_to_pil(output_tensor)
            st.image(output_image_pil, caption="Stylized Output Image", use_column_width=True)

            buf = io.BytesIO()
            output_image_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="stylized_image.png",
                mime="image/png"
            )
    else:
        st.error("Could not load one or both images. Please try again.")

elif stylize_button:
     st.warning("Please upload both Content and Style images.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Upload a 'Content' image.\n"
    "2. Upload a 'Style' image.\n"
    "3. Adjust parameters (optional).\n"
    "4. Click 'Stylize Image'.\n\n"
    "**Note:** Running on CPU can take a very long time. GPU recommended."
)