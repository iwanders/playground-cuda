# Playground CUDA

## Easy introduction

Roughly following https://developer.nvidia.com/blog/even-easier-introduction-cuda/

To reproduce numbers from that article:
```
nsys nvprof ./main
```

## mwc_random_seed

In some games a [mutlipy-with-carry](https://en.wikipedia.org/wiki/Multiply-with-carry_pseudorandom_number_generator) pseudorandom number generator is used. The code here implements a particular set of random rolls in a game such that we can brute force the initialisation seed, and find the one that would result in the numbers observed in the game.

Confirmed this works in the offline game, but the online servers seem to use a different (true random?) random number generator.

The main logic is in the `mwc_cuda.cu` file, the `Makefile` can be used to compile and run this for some tests. When compiled through the `Makefile` it sets a preprocessor variable that enabled compilation of a `main` function, this is omitted when compiled through cargo.

The `mwc_cuda.rs` file shows how to do rust bindings that allow calling into the library compiled with `nvcc`. The unit tests then call into these and verify results. This is modelled after one of the `cudarc` [examples](https://github.com/coreylowman/cudarc/tree/d7ac2b481cb637f7a73f2520ff2d12809285133f/examples/07-build-workflow).

## Background

These two videos are well worth the watch:
- [GTC 2022 - How CUDA Programming Works - Stephen Jones, CUDA Architect, NVIDIA](https://www.youtube.com/watch?v=QQceTDjA4f4)
- [How GPU Computing Works | GTC 2021](https://www.youtube.com/watch?v=3l10o0DYJXg) also by Stephen Jones

# License
License is [Apache](./LICENSE-APACHE) or [MIT](./LICENSE-MIT).
