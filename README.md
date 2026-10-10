# Dress Up for Windows

Realistic virtual try-on for the ASUS ROG Ally X (Windows 10/11 x64). Your photo on the left, a garment on the right,
press **TRY ON**. The old clothes are removed and the new garment is **generated** by a diffusion try-on model
(CatVTON), so it drapes with folds, light and fit, instead of being pasted on. Face, hair, hijab/scarf, hands,
shoes, bag and the background are copied back pixel-for-pixel; only the clothing area is regenerated.

## Run
1. Unzip `DressUp-1.0.0-win64.zip`, open the `DressUp` folder, run **DressUp.exe**.
2. SmartScreen (unsigned): **More info -> Run anyway**.
3. First Try On downloads the models once (~1.9 GB, resumable, SHA-256 checked) to `%LOCALAPPDATA%\DressUp\models`
   from the [models-v1 release](https://github.com/vanukrishnans-source/dress-up-windows/releases/tag/models-v1).

## Use (one fixed screen, no scrolling)
- Left: your photo (front-facing; full body for dresses). Right: the garment (product photo, flat lay, or someone wearing it).
- Garment: **Upper / Lower / Full dress**. Quality: **Fast** (20 steps, 384x512) or **Best** (30 steps, 576x768).
- **TRY ON**, progress + **Cancel**, tap the result or **Before / After**, **Save PNG / Save JPEG** (full original
  resolution) to `Pictures\DressUp`.

## How it works
1. SegFormer-B2 clothes parser finds the current clothes, arms, legs, face, hair, scarf, hands, shoes, bag.
2. Mask = clothes of the chosen type (+ arms for tops/dresses, legs for bottoms/dresses so sleeve/hem length can
   change), closed and dilated. Face, hair, scarf/hijab, hands, shoes and bag are always protected.
3. A 3:4 crop around the mask goes through CatVTON (SD1.5-inpainting UNet with CatVTON attention, garment image
   concatenated as the condition, DDIM, CFG 2.5). The masked area is always filled with the garment.
4. The generated crop is scaled back and blended only inside the mask; every other pixel is the original.

## GPU / CPU
The try-on runs in a separate worker process (`DressUp_cli.exe`). DirectML (torch-directml) is probed in a child
process first; if the Radeon driver fails or crashes, the app falls back to CPU and the window stays open.
Rough timings (Ally X, Z1 Extreme 8 cores): CPU about 1-1.5 min Fast, 5-7 min Best; 780M DirectML best guess
2-4x faster (not measured).

## Safety
Always-on NSFW filter (Falconsai ViT): person photo, garment photo and the result; blocked above 0.85 (Relaxed).
No off switch. CatVTON uses no text prompt (so there are no prompts, positive or negative); the app can only fill
the clothing area with the chosen garment, and refuses photos where no clothing is found.

## License
Personal, non-commercial use (CatVTON weights are CC BY-NC-SA 4.0). See THIRD_PARTY.md.

## Known limits (1.0.0)
- Tops on a person wearing a one-piece outfit with a hijab/scarf over the chest (e.g. a T-shirt onto a long dress) come out weak: the top is changed, but colour/print don't follow the garment reliably. Full dresses work well. Best results: swap like-for-like (top onto a top, dress onto a dress), front-facing photo.
- Garment photos of a person wearing the item are cut out automatically (cloth segmentation); flat product photos work best.
