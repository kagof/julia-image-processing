# Image Processing in Julia

This repository contains a collection of image processing algorithms written in Julia. Some of them take advantage of the CUDA module to run on an NVidia GPU.

Currently there exists:

## Kuwahara

An implementation of the basic [Kuwahara filter](https://en.wikipedia.org/wiki/Kuwahara_filter). This filter creates a an oil painting-esque look. I've not (yet?) implemented more advanced versions of the Kuwahara filter (like the Generalized version. The traditional version does have a blocky appearance and doesn't preserve edges very well.

![Example application of Kuwahara filter](/examples/mandrill_kuwahara.png)

The results are deterministic, although the GPU and CPU versions' outputs vary slightly as can be seen below:

![Difference between GPU and CPU versions](/examples/mandrill_kuwahara_difference.png)

### [Kuwahara CPU](/src/Kuwahara.jl)

This takes advantage of however many threads are available to Julia.

```julia
kuwahara(image_in, region_size::Int=13)
```

### [Kuwahara GPU](/src/KuwaharaGpu.jl)

This uses an NVidia GPU to dramatically speed up the process. The difference in performance is especially seen with larger region sizes. The code took much more effort as it seems that many standard functions (such as `sum`, `map`, `std`, etc) cannot be used in a kernel, and thus have to essentially be re-invented.

```julia
kuwaharaGpu(image_in, 
    region_size::Int = 13, 
    threads_per_block::Int = 256)
```

## Testing

No unit tests written yet, however there is [a util](/src/Test.jl) to play around with these processors and display the output side-by-side with the input.

```julia
testKuwahara(img_name::String="mandrill"; 
    region_size::Int=13, 
    use_gpu::Bool=false, 
    threads_per_block::Int=512)
```