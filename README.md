### Cite Us

[Link to the Research Paper (final version)](https://doi.org/10.1007/s11042-025-20990-0).

If you find our work useful in your research, please cite us:

```
@article{srinivasan2025,
  doi = {10.1007/s11042-025-20990-0},
  url = {https://doi.org/10.1007/s11042-025-20990-0},  
  title={Text-conditioned image generation using diffusion models},
  author={Srinivasan, Dhanya and Mirunalini, P and Desingu, Karthik and MR, Maheshwari},
  journal={Multimedia Tools and Applications},
  pages={1--17},
  year={2025},
  publisher={Springer}
}
```

### Dataset sources
- [MUSTI: Multimodal Understanding of Smells in Texts and Images](https://github.com/multimediaeval/2023-MUSTI-Task/tree/main)
- [COCO: Common Objects in Context](https://cocodataset.org/#home)   

### Hardware and software platform used
- NVIDIA GeForce GTX 1080 Ti GPU
- CUDA Version 12.3
- INTEL XEON(R) CPU E5-1650 v4 @ 3.60 GHz x12

### Code execution order
1. `train_vqvae.py`
2. `train_pixelsnail.py` (for both, top and bottom configurations)
3. `testing.py`


