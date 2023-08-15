# Accelerating the Convergence of Physics-Informed Neural Networks Using New Training Methods and Backbone Networks

- Author：LiangLiang Yan



## Abstract

While traditional numerical methods have achieved high precision by discretely solving Partial Differential Equations (PDEs), they grapple with the curse of dimensionality and often falter in tackling high-dimensional problems. Physics-Informed Neural Networks (PINNs) have demonstrated tremendous potential and robustness in solving PDEs, but their training often suffers from instability. However, traditional PINNs (vanilla PINNs) are based predominantly on fully connected neural networks (FC), an architecture that exhibits issues pertaining to convergence difficulty and parameter redundancy. In this paper, we introduce an innovative PINNs based on a backbone network reminiscent of a multi-input residual network(MultiIn-net PINNs). Specifically, we leverage a multi-step training paradigm to facilitate PINNs training without any supervised data. Our experiments reveal that MultiIn-net PINNs can maintain better convergence with a leaner parameter set compared to other backbone networks(e.g., fully connected neural network (FC), residual neural network(Resnet), UNet). The multi-step training methodology augments convergence speed by approximately 45% compared to vanilla PINNs in our examples, while multiIn-net enhances it by around 50%, together, that's an increase of 70%.

---







## Citation

```bash
##
```

