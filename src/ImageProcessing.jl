module ImageProcessing

using Images, ImageView, TestImages, Statistics, CUDA

export testKuwahara, kuwaharaGpu, kuwahara

include("Kuwahara.jl")
include("KuwaharaGpu.jl")
include("test.jl")

end # module