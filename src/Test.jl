function testKuwahara(img_name::String="mandrill"; region_size::Int=13, use_gpu::Bool=false, threads_per_block::Int=512)
    img = testimage(img_name)
    @assert !(img isa Nothing)
    kuwa = @time "total" use_gpu ? kuwaharaGpu(img, region_size, threads_per_block) : kuwahara(img, region_size)
    mosaicview(img, kuwa; nrow=1)
end