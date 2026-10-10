# Third-party components

| Component | Use | License |
|---|---|---|
| CatVTON (Zheng-Chong/CatVTON, `mix-48k-1024` attention weights) | try-on diffusion model | CC BY-NC-SA 4.0 (non-commercial) |
| Stable Diffusion 1.5 inpainting UNet (runwayml, mirror booksforcharlie/stable-diffusion-inpainting) | base of the try-on model | CreativeML OpenRAIL-M |
| sd-vae-ft-mse (Stability AI) | image encoder/decoder | MIT |
| SegFormer B2 clothes (mattmdjaga/segformer_b2_clothes), exported to ONNX | finds the clothes to replace | MIT |
| Falconsai nsfw_image_detection (ViT, int8 ONNX) | always-on safety filter | Apache-2.0 |
| PyTorch, torch-directml, diffusers, ONNX Runtime, OpenCV, Pillow, NumPy, PySide6 (LGPL-3.0) | runtime | BSD / MIT / Apache-2.0 / LGPL |

The merged model file `catvton_unet_fp16.safetensors` = SD1.5-inpainting UNet with CatVTON's self-attention weights,
cross-attention removed, stored fp16. It carries the CatVTON CC BY-NC-SA 4.0 and OpenRAIL-M use restrictions.
Test photos in `testdata/`: Wikimedia Commons, CC0 / public domain.
