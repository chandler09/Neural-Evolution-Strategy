# NeurIPS25': Neural Evolution Strategy for Black-box Pareto Set Learning

<img width="1504" height="365" alt="image" src="https://github.com/user-attachments/assets/ab5eb970-f3d1-4b95-be68-6fbd155baa2a" />

Neural Evolution Strategy (Neural-ES) uses a feed-forward neural network to learn a Pareto-Set (PS) manifold and, at the same time, model the search distribution of an Evolution Strategy (ES). By estimating the natural gradient, as most ES algorithms do, it can learn the PS manifold without gradient information from the problem. The black-box PS learning paradigm differs from traditional evolutionary multi-objective optimization by enabling zero-shot optimization for unseen user preferences. For more details, you are encouraged to read our [paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa7b618cb7f8b35ba06e9418c2cd1c1c-Paper-Conference.pdf).  

The code serves as an official implementation of Neural-ES. We keep a minimal dependency on Python packages, which include: 
- Pytorch (and CUDA, of course)
- Numpy
- [Pymoo](https://pymoo.org/) (for multi-objective optimization)

You may find a minimal usage example here. 

We would appreciate it if you found our work helpful and would kindly cite our paper: 
```bash
@article{lu2026neural,
  title={Neural Evolution Strategy for Black-box Pareto Set Learning},
  author={Lu, Chengyu and Li, Zhenhua and Lin, Xi and Cheng, Ji and Zhang, Qingfu},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={171463--171495},
  year={2026}
}
```

Enjoy and feel free to reach out at `chengyulu3-c@my.cityu.edu.hk` :)
