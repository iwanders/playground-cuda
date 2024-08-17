# Playground CUDA

## Easy introduction

Roughly following https://developer.nvidia.com/blog/even-easier-introduction-cuda/

To reproduce numbers from that article:
```
nsys nvprof ./main
```

## mwc_random_seed

In some games a [mutlipy-with-carry](https://en.wikipedia.org/wiki/Multiply-with-carry_pseudorandom_number_generator) pseudorandom number generator is used. The code here implements a particular set of random rolls in a game such that we can brute force the initialisation seed based on the visible observations in the game. Confirmed this works in the offline game, but the online servers seem to use a different (true random?) random number generator.

## Background

These two videos are well worth the watch:
- [GTC 2022 - How CUDA Programming Works - Stephen Jones, CUDA Architect, NVIDIA](https://www.youtube.com/watch?v=QQceTDjA4f4)
- [How GPU Computing Works | GTC 2021](https://www.youtube.com/watch?v=3l10o0DYJXg) also by Stephen Jones

