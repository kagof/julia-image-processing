function bokehBlurGpu(image_in, aperture=map(Gray, ones(12, 12)), threads_per_block::Int=256)
    image = RGB.(image_in)
    aperture = Gray.(aperture)
    result = CUDA.zeros(size(channelview(float.(image_in)))...)
    total_size = Int32(length(image))

    image_size_y = Int32(size(image_in, 1))
    image_size_x = Int32(size(image_in, 2))

    divisor = Float32(gray(sum(aperture)))
    if (divisor <= 0)
        return result
    end

    mask_size_y = Int32(size(aperture, 1))
    mask_size_x = Int32(size(aperture, 2))

    # deals with odd & even numbered mask sizes
    mask_size_y_half_under = Int32(mask_size_y ÷ 2)
    mask_size_x_half_under = Int32(mask_size_x ÷ 2)

    image_cu = CuArray(Float32.(channelview(image)))
    aperture_cu = CuArray(Float32.(gray.(aperture)))

    function computePixelKernel!(image, aperture, result)
        threadId = threadIdx().x
        pos = (blockIdx().x - 1) * blockDim().x + threadId
        if (pos > total_size)
            return nothing
        end
        y = ((pos - 1) % image_size_y) + 1
        x = ((pos - 1) ÷ image_size_y) + 1

        window_min_y = (y - mask_size_y_half_under)
        window_min_x = (x - mask_size_x_half_under)

        divisor_adjusted = divisor
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0
        i = 1
        while i <= mask_size_y
            j = 1
            while j <= mask_size_x
                px_y = window_min_y + i - 1
                px_x = window_min_x + j - 1
                if (px_y < 1 || px_x < 1 || px_y > image_size_y || px_x > image_size_x)
                    divisor_adjusted -= aperture[i, j]
                else
                    sum_r += aperture[i, j] * image[1, px_y, px_x]
                    sum_g += aperture[i, j] * image[2, px_y, px_x]
                    sum_b += aperture[i, j] * image[3, px_y, px_x]
                end
                j += 1
            end
            i += 1
        end
        result[1, y, x] = min(1, max(0, sum_r / divisor))
        result[2, y, x] = min(1, max(0, sum_g / divisor))
        result[3, y, x] = min(1, max(0, sum_b / divisor))
        return nothing
    end

    threads_actual = min(threads_per_block, total_size)

    CUDA.@time @cuda threads = threads_actual blocks = Int(ceil(total_size / threads_actual)) computePixelKernel!(image_cu, aperture_cu, result)
    return colorview(RGB{N0f8}, N0f8.(Array(result)))

    return result
end