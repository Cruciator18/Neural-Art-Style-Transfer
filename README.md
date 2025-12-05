
# 🎨 PyTorch Implementation: A Neural Algorithm of Artistic Style

This repository contains a PyTorch implementation of the Neural Style Transfer algorithm described in the landmark paper **["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)** by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

The project is deployed as an interactive web application using **Streamlit**, allowing users to generate artistic images on the fly.

## ✨ Features

* **Interactive Web UI:** Seamlessly upload Content and Style images to visualize results directly in your browser.
* **Original Paper Replication:** Utilizes the pre-trained **VGG-19** network architecture as described in the original research.
* **Real-time Parameter Tuning:** Adjust hyperparameters via the sidebar, including:
    * Content vs. Style Weights ($\alpha$ and $\beta$)
    * Output Image Size
    * Optimization Steps
* **Advanced Feature Extraction:**
    * **Content:** Extracted from layer `conv4_2`.
    * **Style:** Gram matrices computed from layers `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1`.
* **Optimization:** Uses the **Adam** optimizer for faster convergence compared to L-BFGS.
* **Smart Device Support:** Automatically detects CUDA-enabled GPUs.
    * *Note:* If running locally with an NVIDIA GPU, inference will be significantly faster.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
````

### 2\. Install Dependencies

Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
pip install torch torchvision numpy pillow streamlit
```

### 3\. Run the Streamlit App

Execute the following command in your terminal to launch the web interface:

```bash
streamlit run app.py
```

The app should automatically open in your default browser at `http://localhost:8501`.

-----

## 🧠 How It Works

The core concept is that Deep Convolutional Neural Networks (CNNs) learn hierarchical representations of images. We use a pre-trained **VGG-19** model to separate and recombine content and style.

### The Two Components

1.  **Content Representation:** Higher layers of the network (like `conv4_2`) capture the complex structure and arrangement of objects.
2.  **Style Representation:** Lower layers capture textures, colors, and brushstrokes. We calculate the **Gram Matrix** (correlations between filter responses) to represent style mathematically.

### The Algorithm

The algorithm starts with a random noise image (or the content image) and iteratively updates its pixel values to minimize a combined loss function:

$$ \mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{content} + \beta \cdot \mathcal{L}_{style} $$

Where:

  * $\mathcal{L}_{content}$ is the Mean Squared Error (MSE) between feature maps of the Generated and Content images.
  * $\mathcal{L}_{style}$ is the MSE between the Gram matrices of the Generated and Style images.

-----

## 🛠 Technical Implementation

### 1\. Model Loading

We load a pre-trained `VGG19` from `torchvision.models`.

  * **Features Only:** The fully connected classification layers are discarded.
  * **Eval Mode:** The model is set to `.eval()` to freeze weights (we optimize the input image, not the network).

### 2\. Loss Calculation

  * **Content Loss:** Calculated at layer `conv4_2`.
  * **Style Loss:** Calculated and summed across layers `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1`.

### 3\. Optimization Loop

  * **Target:** The input image is treated as the trainable parameter.
  * **Optimizer:** We use `torch.optim.Adam`.
  * **Loop:** For $N$ steps, we forward pass the image, calculate loss, and backpropagate to update the pixel values.

-----

## 🙏 Acknowledgments

This implementation is based on the research:

> **A Neural Algorithm of Artistic Style**
> *Leon A. Gatys, Alexander S. Ecker, Matthias Bethge*
> [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)

