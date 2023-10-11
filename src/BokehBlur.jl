function bokehBlur(image_in, aperture=map(Gray, ones(12, 12)))
    image = RGB.(image_in)
    aperture = Gray.(aperture)
    result = zeros(RGB{N0f8}, size(image)...)

    image_size = size(image)

    divisor = sum(aperture)
    if (divisor <= 0)
        return result
    end

    mask_size = size(aperture)

    # deals with odd & even numbered mask sizes
    mask_size_y_half_under = mask_size[1] ÷ 2
    mask_size_y_half_over = Int(ceil(mask_size[1] / 2))
    mask_size_x_half_under = mask_size[2] ÷ 2
    mask_size_x_half_over = Int(ceil(mask_size[2] / 2))

    function computePixel(y, x)
        window_y = (y-mask_size_y_half_under):(y+mask_size_y_half_over)
        window_x = (x-mask_size_x_half_under):(x+mask_size_x_half_over)
        divisor_adjusted = divisor
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0
        for i in 1:mask_size[1]
            for j in 1:mask_size[2]
                px_y = window_y[i]
                px_x = window_x[j]
                if (px_y < 1 || px_x < 1 || px_y > image_size[1] || px_x > image_size[2])
                    divisor_adjusted -= aperture[i, j]
                else
                    sum_r += gray(aperture[i, j]) * image[px_y, px_x].r
                    sum_g += gray(aperture[i, j]) * image[px_y, px_x].g
                    sum_b += gray(aperture[i, j]) * image[px_y, px_x].b
                end
            end
        end
        result[y, x] = RGB(map(v -> N0f8(min(1, max(0, v / divisor))), [sum_r, sum_g, sum_b])...)
    end

    @time "computation" Threads.@threads for y in 1:size(image, 1)
        for x in 1:size(image, 2)
            computePixel(y, x)
        end
    end

    return result
end