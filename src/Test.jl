function testKuwahara(img_name::String="mandrill"; region_size::Int=13, use_gpu::Bool=false, threads_per_block::Int=512)
    img = testimage(img_name)
    @assert !(img isa Nothing)
    kuwa = @time "total" use_gpu ? kuwaharaGpu(img, region_size, threads_per_block) : kuwahara(img, region_size)
    mosaicview(img, kuwa; nrow=1)
end

function testBoxBlur(img_name::String="mandrill"; box_size::Int=12, use_gpu=false, threads_per_block::Int=512)
    img = testimage(img_name)
    @assert !(img isa Nothing)
    box = @time "total" use_gpu ? bokehBlurGpu(img, ones(Gray, box_size, box_size), threads_per_block) : bokehBlur(img, ones(Gray, box_size, box_size))
    mosaicview(img, box; nrow=1)
end

function testCircleBokehBlur(img_name::String="mandrill"; use_gpu=false, threads_per_block::Int=512)
    img = testimage(img_name)
    @assert !(img isa Nothing)
    circle = @time "total" use_gpu ? bokehBlurGpu(img, load("examples/aperture.bmp"), threads_per_block) : bokehBlur(img, load("examples/aperture.bmp"))
    mosaicview(img, circle; nrow=1)
end

function testStarBokehBlur(img_name::String="mandrill"; use_gpu=false, threads_per_block::Int=512)
    img = testimage(img_name)
    @assert !(img isa Nothing)
    star = @time "total" use_gpu ? bokehBlurGpu(img, load("examples/staraperture.bmp"), threads_per_block) : bokehBlur(img, load("examples/staraperture.bmp"))
    mosaicview(img, star; nrow=1)
end

function testGaussianBlur(img_name::String="mandrill"; iterations::Int=1, use_gpu=false, threads_per_block::Int=512)
    img = testimage(img_name)
    @assert !(img isa Nothing)
    threex3 = img
    fivex5 = img
    @time "total" for _ in 1:iterations
        threex3 = use_gpu ? bokehBlurGpu(threex3, [1 2 1; 2 4 2; 1 2 1] / 16, threads_per_block) : bokehBlur(threex3, [1 2 1; 2 4 2; 1 2 1] / 16)
        fivex5 = use_gpu ? bokehBlurGpu(fivex5, [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1] / 256, threads_per_block) : bokehBlur(fivex5, [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1] / 256)
    end
    mosaicview(img, threex3, threex3 - fivex5, fivex5; nrow=2)
end