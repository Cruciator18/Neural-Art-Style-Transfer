import torch
import torch.nn as nn
import torchvision.transforms as transform
from PIL import Image
import torchvision.models as models
import os
import urllib.request
import torch.nn.functional as F

IMAGE_SIZE = 256
ALPHA = 1
BETA = 1e7
NUM_STEPS = 1000
LEARNING_RATE = 0.015

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

# NORMALIZATION AND TRANSFORMATIONS FOR THE INPUT IMAGES 

normalized_mean = torch.tensor([0.485, 0.456 , 0.406]).to(device)
normalized_std =  torch.tensor([0.429 , 0.224 , 0.225]).to(device)

transforms = transform.Compose([
    transform.Resize((IMAGE_SIZE,IMAGE_SIZE),),
    transform.ToTensor(),
    transform.Normalize(mean=normalized_mean , std = normalized_std)
    
])

# LOADING THE IMAGE

def _load_image(image_name):
    
    image = Image.open(image_name).convert('RGB')
    image = transforms(image).unsqueeze(0)
    return image.to(device, torch.float)


def _save_image(tensor , filename):
    
    image = tensor.cpu().clone().squeeze()
    
    mean_cpu = normalized_mean.cpu() 
    std_cpu = normalized_std.cpu()
    
    image =  image * std_cpu.view(3,1,1) + mean_cpu.view(3,1,1)
    image = image.clamp(0,1)
    
    
    image = transform.ToPILImage()(image)
    image.save(filename)
    print(f"Image saved as {filename}")

    

class VGGFeatureExtractor(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        
        vgg_model = models.vgg19(weights = True)
        vgg_features = vgg_model.features.to(device)
        
        
        vgg_features.eval()
        
        for params in vgg_features.parameters():
            
            params.requires_grad_(False)
        
        
        self.content_layers = ['21']
        self.style_layers = ['0', '5', '10', '19', '28']  
        
        all_layers = [int (l) for l in self.content_layers + self.style_layers]
        max_layers_idx = max(all_layers)
        
        self.features = nn.Sequential(*[vgg_features[i] for i in range(max_layers_idx + 1)])  
        
        
    def forward(self , x):
        
        content_features_out = {}
        style_features_out = {}
        
        
        for name , layer in self.features._modules.items():
            
            x = layer(x)
            
            if name in self.content_layers:
                content_features_out['content'] = x
                
            if name in self.style_layers:
                style_features_out[name] = x
                
        return content_features_out , style_features_out                





def gram_matrix(input_tensor):
        
        B , C, H, W = input_tensor.size()
        
        features = input_tensor.view(C , H*W)
        
        G = torch.mm(features , features.t())
        
        return G.div(C*H*W)
    
    
if __name__ == "__main__":
    
    content_filename = "content.jpg"
    style_filename = "style.jpg"
    
        
          
    print("Loading images and model...")
    content_image = _load_image(content_filename)
    style_image = _load_image(style_filename)
    
    
    model = VGGFeatureExtractor()
    
    
    target_content_features , _ =model(content_image)
    target_content_tensor =  target_content_features['content']
    
    
    _, target_style_features = model(style_image)
    target_style_grams = {name : gram_matrix(feat) for name,feat in target_style_features.items()}
    
    
    
    target_image = content_image.clone().requires_grad_(True)
    
    
    optimizer = torch.optim.Adam([target_image] , lr = LEARNING_RATE)
    
    print("Starting style transfer optimization...")
    
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        
        current_content_features , current_style_features = model(target_image)
        
        content_loss = F.mse_loss(current_content_features['content'] , target_content_tensor)
        
        style_loss = 0
        
        for layer_name in current_style_features:
            current_gram = gram_matrix(current_style_features[layer_name])
            target_gram = target_style_grams[layer_name]
            style_loss += F.mse_loss(current_gram , target_gram)
        
        
        
        total_loss =  ALPHA*content_loss + BETA*style_loss
        
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}/{NUM_STEPS} | Total Loss: {total_loss.item():.4f} | "
                  f"Content Loss: {content_loss.item():.4f} | Style Loss: {style_loss.item():.4f}")

    # --- Save the final image ---
    print("Optimization finished.")
    _save_image(target_image, "output1.jpg")  




    
    

