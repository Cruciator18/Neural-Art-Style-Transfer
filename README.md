 ## 🎨 PyTorch Implementation of "A Neural Algorithm of Artistic Style"
This repository contains a PyTorch implementation of the neural style transfer algorithm described in the paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

This implementation is served as an interactive web application using Streamlit.

 ## Features

### Interactive Web UI:
Easily upload content and style images and see results directly in your browser.

 ### Replicates the Original Paper: 
 Uses a pre-trained VGG-19 network.

### Real-time Parameter Tuning:
Adjust content/style weights, image size, and optimization steps from the UI.

### Feature Extraction:
Uses content features from conv4_2 and style features (Gram matrices) from five layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1).

### Adam Optimizer:
Uses the Adam optimizer to generate the target image.

### Device Support:
Automatically detects and uses a CUDA-enabled GPU or falls back to CPU. (Though the streamlit app uses only CPU , you can make your local inferences using your GPU)

# 🧠 How It Works

### The core idea of the paper is that deep Convolutional Neural Networks (CNNs), like VGG, learn hierarchical feature representations of images.

Content: The higher layers of the network capture the high-level content of an image (i.e., what objects are in the image and their arrangement).

Style: The lower layers capture the style of an image (i.e., how the image is painted—the texture, colors, and local patterns). The paper proposes using a Gram matrix to represent this style as the correlation between different filter responses.

The algorithm works by generating a new image that simultaneously minimizes two losses:

Content Loss: The distance (MSE) between the content features of the generated image and the original content image.

Style Loss: The distance (MSE) between the style representations (Gram matrices) of the generated image and the original style image.

This process is broken down as follows:

 ## 1. Load Model

A pre-trained VGG-19 network is loaded from torchvision.models. We only need its features, so the classifier (fully connected) layers are discarded and the model is set to evaluation mode (.eval()).

## 2. Define Losses

Content Loss: The L2-norm (Mean Squared Error) between the feature maps of the content image and the generated image at a specific intermediate layer (conv4_2).

 Style Loss: The L2-norm (Mean Squared Error) between the Gram matrices of the feature maps of the style image and the generated image. This is calculated at multiple layers (conv1_1...conv5_1) and summed.

## 3. Optimization

A target image is initialized as a clone of the content image.

This target image is treated as the only trainable parameter.

The Adam optimizer iteratively updates the pixels of the target image to minimize the total loss:

`` total_loss = (alpha * content_loss) + (beta * style_loss) ``

 ## 4. Result

After many iterations, the target image visually retains the content of the content image while adopting the texture and color palette of the style image.

 # 🙏 Acknowledgments

This code is based on the original paper: A Neural Algorithm of Artistic Style by Gatys, Ecker, and Bethge.


