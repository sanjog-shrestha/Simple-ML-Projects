# ============================================================
# Black & White Image Colorization (Colab) — PyTorch version
# Uses the official richzhang/colorization repo (actively maintained)
# ============================================================

# 1. Install dependencies
!pip install -q torch torchvision scikit-image matplotlib

# 2. Clone the official repo (contains model code + demo images)
!git clone -q https://github.com/richzhang/colorization.git
%cd colorization

# 3. Imports
import matplotlib.pyplot as plt
from colorizers import *

# 4. Load pretrained colorizers (weights auto-download from S3 on first run)
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

use_gpu = False  # set True if you enabled a GPU runtime
if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

# 5. Load an image — uses a sample from the repo by default
img_path = 'imgs/ansel_adams3.jpg'   # replace with your own uploaded filename

img = load_img(img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
if use_gpu:
    tens_l_rs = tens_l_rs.cuda()

# 6. Run both models
img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

# 7. Display results
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1); plt.imshow(img); plt.title('Original'); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(img_bw); plt.title('Grayscale input'); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(out_img_eccv16); plt.title('ECCV16 output'); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(out_img_siggraph17); plt.title('SIGGRAPH17 output'); plt.axis('off')
plt.show()