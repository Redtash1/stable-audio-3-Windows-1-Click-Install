<div align="center">


# DON'T CLONE THIS REPO, IT WON'T WORK AS IT ALL DEPENDS ON THE PYTHON_EMBEDED 3.12.10 TO WORK! 


## I made this Stable Audio 3 Portable 1 click install for Windows that uses Nvidia GTX 10XX, 16XX, RTX Quadro, 20XX, 30XX, 40XX, 50XX GPU. Installs Torch with Cuda, Flash Attention, Triton & all other requirements also creates Launch Stable Audio 3 & Sounds Desktop Shortcuts. Automatically downloads models into "ckpts" folder upon the first use of each of the 3 different models. Automatically saves generations into "outputs" folder with your choice of .wav or .mp3. Automatically saves last used settings in .json file & loads them automatically on startup.


## Click here to jump to Install 👉 [Installation](#-Installation) 👈


![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/redtash1/stable-audio-3-Windows-1-Click-Install/total?style=for-the-badge&labelColor=orange&color=0000ff)



</div> 


---


## Simple Tab
<img width="1828" height="941" alt="Screenshot 2026-06-23 152725" src="https://github.com/user-attachments/assets/82274196-697c-4663-8906-d449d7d3f247" />


----


## Advance Tab with Inpainting
<img width="1819" height="956" alt="Screenshot 2026-06-23 152844" src="https://github.com/user-attachments/assets/3ca8a456-e38b-4135-90f7-5b29f371aed5" />


----


# Stable Audio 3

**A state-of-the-art open platform for fast, high-quality generated audio and music.**

[Technical Report](https://arxiv.org/abs/2605.17991) · [🤗 Models](https://huggingface.co/collections/stabilityai/stable-audio-3) ·  · [Discord](https://discord.gg/7QM7mtY9uH) · [Demo](https://huggingface.co/spaces/stabilityai/stable-audio-3) · [Blog Post](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models)


Stable Audio 3 is the next generation of Stable Audio: a focused, streamlined platform for inference and fine-tuning, built on lessons from [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). If you're doing foundational research or working with previous Stable Audio models, that repo is still the place to go.


---


## Models

| Model | Model ID | Autoencoder | Hardware | Params | Max length | Use case |
|---|---|---|---|---|---|---|
| [**Stable Audio 3 Small-Music**](https://huggingface.co/stabilityai/stable-audio-3-small-music) | `small-music` | SAME-Small | CPU | 433M | 120s | Lightweight music-only inference, no GPU required |
| [**Stable Audio 3 Small-SFX**](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) | `small-sfx` | SAME-Small | CPU | 433M | 120s | Lightweight sound effects-only inference, no GPU required |
| [**Stable Audio 3 Medium**](https://huggingface.co/stabilityai/stable-audio-3-medium) | `medium` | SAME-Large | GPU (CUDA) | 1.4B | 380s | High Quality, Fast Inference |
| **Stable Audio 3 Large** | — | SAME-Large | API only | 2.7B | 380s | Highest quality, API only. Not supported by this repo, see the [API docs](https://platform.stability.ai/docs/api-reference#tag/Stable-Audio) |


### Performance

| Model | Duration | H200 | H200 + TensorRT | Mac CPU* | Mac CoreML | Peak VRAM† |
|---|---|---|---|---|---|---|
| `small` | 5s | 0.41s | 0.017s | 0.70s | 0.23s | 1.69 GB |
| `small` | 30s | 0.46s | 0.022s | 1.72s | 0.63s | 1.89 GB |
| `small` | 120s | 0.45s | 0.044s | 5.92s | 3.09s | 2.40 GB |
| `medium` | 5s | 0.60s | 0.02s | – | – | 5.07 GB |
| `medium` | 30s | 0.65s | 0.05s | – | – | 5.49 GB |
| `medium` | 120s | 0.78s | 0.13s | – | – | 6.49 GB |
| `medium` | 380s | 1.31s | 0.43s | – | – | 6.52 GB |

\* CPU-only via CoreML (Diffusion Transformer) + TFLite (SAME-S decoder)
† Peak allocated VRAM on H200, unchunked decode. Chunked decoding reduces this — e.g. `medium` at 120s drops from 6.49 GB to ~5.14 GB.

---

## Features
- ⚡ **Fast, state-of-the-art generation** - Generate minutes of audio in milliseconds
- 🎛️ **Three inference modes** — text-to-audio, audio-to-audio editing, and inpainting/continuation
- ↔️ **Variable-length generation** — handles generation of a variety of sequences without wasting inference time and VRAM on unused latents
- 🎯 **Personalization through LoRA fine-tuning** — adapt any model to a target style; stackable, adjustable at runtime
- 🎵 **SAME autoencoder** — new Semantic-Acoustic Music Encoder; stereo, 44.1 kHz, 256-dimensional latents optimized for both generative tractability and high-quality reconstruction


# 📦 Installation

## Nvidia GTX 10XX, 16XX, RTX Quadro, 20XX, 30XX, 40XX, 50XX  

## GTX 10XX-RTX 30XX will have torch 2.6.0+cu126 installed for compatibility. RTX 40XX & 50XX will have torch 2.7.1+cu128.

## Small - Text to audio models requires 4GB VRAM. 
## Medium - Text to audio model  requires  requires 8GB - 12GB VRAM. Older GPU's may not be able to use the medium model as it requires Flash Attention 2.


1. Make sure you have Git installed, if not download the Git Standalone Installer and click on Git for Windows/x64 Setup. 👉 [Git Standalone Installer Download](https://git-scm.com/downloads/win) 👈 To install Git, double click Git.exe and just keep clicking next until it's installed, you don't need to change anything.


2. Make sure your Nvidia graphics drivers are up-to-date. If they are not or if your not sure, please click on the following link to download Nvidia graphics drivers. 👉 [Nvidia Drivers](https://www.nvidia.com/en-us/software/nvidia-app/) 👈

3. Make sure that you have NVIDIA's [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) version **12.8** (or newer) installed on your system.

4. Make sure you have FFMPEG Shared downloaded & on PATH. Download 👉 [ffmpeg-release-full-shared.7z](https://www.gyan.dev/ffmpeg/builds/) 👈

5.  Now after you have made sure Nvidia GPU drivers are up to date and Git is installed, download Stable Audio 3 Windows 1 Click Install
 from here 👉 [Stable Audio 3 Windows 1 Click Install](https://github.com/Redtash1/stable-audio-3-Windows-1-Click-Install/releases) 👈 or from the Releases section at the top right of this page.

6. After downloading, extract Stable Audio 3 Windows 1 Click Install ZIP file and pick where you would like to extract the zip files too.

7. Then open Stable Audio 3 Windows 1 Click Install main folder, you will see this in the root
----

<img width="515" height="156" alt="Screenshot 2026-06-23 142811" src="https://github.com/user-attachments/assets/4f3f5541-5ec0-4e32-8285-32b85672e209" />

----
8. Then depending on your Nvidia GPU double click on either the Install_Stable_Audio_3__GTX_10XX_RTX_20XX_30XX.bat or Install_Stable_Audio_3_RTX_40XX_&_50XX.bat to start the installation. It will install everything.  After installation is finished, slowly scroll back up to the top to make sure everything installed correctly.

9. To launch Stable Audio 3 double click the Launch_Stable_Audio_3.bat & it will automatically open in your default Internet Browser.

---

## Troubleshooting

If you have problems after a successful installation, please go to the Official Stable Audio Hugging Face Repo to report problems. [Stable Audio 3](https://huggingface.co/spaces/stabilityai/stable-audio-3/discussions). Thank you.

### If this worked for you, Please give it a Star ⭐. Thank you.

---

## Docs

| Guide | Description |
|-------|-------------|

| [Prompting Guide](docs/guides/prompting.md) | Prompt and control signal reference |

| [Model Overview](docs/guides/model-overview.md) | Architecture and design overview |


---

## Community

- [Harmonai Discord](https://discord.gg/7QM7mtY9uH): Check out our Harmonai Discord server run by the research team. Besides good discussions, we host weekly office hours talking all things AI audio and music and want to hear what you come up with!


- [Awesome Stable Audio](https://github.com/Stability-AI/Awesome-Stable-Audio): Curated list of all community-built Stable Audio projects. Includes links to ComfyUI, Fal, as well as a growing list of community integrations and extensions. 

---


## License

Please refer to the [Stability AI Community License](https://stability.ai/license)

---

## Citation

For Stable Audio 3, please cite
```BibTeX
@misc{evans2026stableaudio3,
  title={Stable Audio 3},
  author={Zach Evans and Julian D. Parker and Matthew Rice and CJ Carr and Zack Zukowski and Josiah Taylor and Jordi Pons},
  year={2026},
  eprint={2605.17991},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2605.17991}
}
```

For SAME, please cite
```BibTeX
@misc{parker2026SAME,
  title={SAME: A Semantically-Aligned Music Autoencoder},
  author={Julian D. Parker and Zach Evans and CJ Carr and Zack Zukowski and Josiah Taylor and Matthew Rice and Jordi Pons},
  year={2026},
  eprint={2605.18613},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2605.18613}
}
```
