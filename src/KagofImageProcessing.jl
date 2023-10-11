module ImageProcessing

using Images, ImageView, TestImages, Statistics, CUDA

export testKuwahara, testBoxBlur, testCircleBokehBlur, testStarBokehBlur, testGaussianBlur, kuwaharaGpu, kuwahara, bokehBlur, bokehBlurGpu

include("Kuwahara.jl")
include("KuwaharaGpu.jl")
include("BokehBlur.jl")
include("BokehBlurGpu.jl")
include("Test.jl")

end # module