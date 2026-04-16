# nanograd

A minimalistic autodiff engine built using only one primitive.

Inspired by: a) ["All elementary functions from a single operator"](https://arxiv.org/html/2603.21852v2) and b) the [micrograd](https://github.com/karpathy/micrograd) engine by Andrey Karpathy

> **Note on performance**
>
> This implementation is intentionally minimal and built around a single primitive.
> As a result, it is not optimized for performance and can be significantly slower
> than conventional autodiff frameworks.
> This implementation prioritizes minimalism and theoretical interest over runtime
> efficiency.
