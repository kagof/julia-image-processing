function kuwaharaGpu(image_in, region_size::Int = 13, threads_per_block::Int = 256)
    region_size::Int32 = Int32(region_size)
    image = RGB.(image_in) |> cu
    size_y = Int32(size(image, 1))
    size_x = Int32(size(image, 2))
    total_size = Int32(length(image))
    brightnesses = channelview(float.(HSV.(image_in)))[3,:,:] |> cu
    result = CUDA.zeros(size(channelview(float.(image_in)))...)
    
    quadrant_size = Int32(ceil(region_size / 2))

    function computePixelKernel!(image, quadrants, brightnesses, result)
        threadId = threadIdx().x
        pos = (blockIdx().x - 1) * blockDim().x + threadId
        if (pos > total_size)
            return nothing
        end
        y = ((pos - 1) % size_y) + 1
        x = ((pos - 1) ÷ size_y) + 1

        top = y - (region_size ÷ 2)
        left = x - (region_size ÷ 2)

        std_1 = Inf
        std_2 = Inf
        std_3 = Inf
        std_4 = Inf

        for q in 1:4
            # quadrants go clockwise from top left
            qtop = top + (quadrant_size * (q > 2))
            qleft = left + (quadrant_size * ((q - 1) % 2))
            for xy in 1:2
                start = (qtop * (xy == 1)) + (qleft * (xy == 2))
                max = (size_y * (xy == 1)) + (size_x * (xy == 2))
                for i in 1:quadrant_size
                    # puts the x/y position in the array, or 0 if it is out of bounds of the image
                    quadrants[pos, q, xy, i] = ((start + i - 1) > 0) * ((start + i - 1) <= max) * (start + i - 1)
                end
            end
            sum_bright = 0.0
            count_bright = 0
            avg_bright = 0.0
            # computing the average brightness
            for i in 1:quadrant_size
                for j in 1:quadrant_size
                    (quadrants[pos, q, 1, i] > 0) && (quadrants[pos, q, 2, j] > 0) && (sum_bright += brightnesses[quadrants[pos, q, 1, i], quadrants[pos, q, 2, j]]; count_bright += 1)
                end
            end
            count_bright > 0 && (avg_bright = sum_bright / count_bright)
            sum_terms = 0
            # using the average brightness to compute the standard deviation
            for i in 1:quadrant_size
                for j in 1:quadrant_size
                    (quadrants[pos, q, 1, i] > 0) && (quadrants[pos, q, 2, j] > 0) && (sum_terms += (brightnesses[quadrants[pos, q, 1, i], quadrants[pos, q, 2, j]] - avg_bright) ^ 2)
                end
            end

            # avoiding conditionals; probably not necessary as the compiler would take care of it anyways
            (q == 1) && (count_bright > 0) && (std_1 = sqrt(sum_terms / count_bright))
            (q == 2) && (count_bright > 0) && (std_2 = sqrt(sum_terms / count_bright))
            (q == 3) && (count_bright > 0) && (std_3 = sqrt(sum_terms / count_bright))
            (q == 4) && (count_bright > 0) && (std_4 = sqrt(sum_terms / count_bright))
        end

        # choose the quadrant with the lowest standard deviation in brightness
        min_std = min(std_1, std_2, std_3, std_4)
        winning_quadrant=1
        (min_std == std_1) && (winning_quadrant = 1)
        (min_std == std_2) && (winning_quadrant = 2)
        (min_std == std_3) && (winning_quadrant = 3)
        (min_std == std_4) && (winning_quadrant = 4)
        
        # computing the average pixel color in the winning quadrant
        sum_r = zero(result[1, 1, 1])
        sum_g = zero(result[1, 1, 1])
        sum_b = zero(result[1, 1, 1])
        count_vals = 0
        for i in 1:quadrant_size
            for j in 1:quadrant_size
                if ((quadrants[pos, winning_quadrant, 1, i] > 0) && (quadrants[pos, winning_quadrant, 2, j] > 0))
                    px = image[quadrants[pos, winning_quadrant, 1, i], quadrants[pos, winning_quadrant, 2, j]]
                    sum_r += px.r
                    sum_g += px.g
                    sum_b += px.b
                    count_vals += 1
                end
            end
        end

        # setting the resulting pixel
        if (count_vals > 0)
            avg_r = sum_r / count_vals
            avg_g = sum_g / count_vals
            avg_b = sum_b / count_vals
            result[1, y, x] = avg_r
            result[2, y, x] = avg_g
            result[3, y, x] = avg_b
        end
        
        return nothing
    end

    
    # There's going to be a much more space & time efficient way to do this, quadrants has one block of memory per pixel, with space for 4 quadrants worth of x and y coordinates
    quadrants = CUDA.zeros(Int32, total_size, 4, 2, quadrant_size)

    threads_actual = min(threads_per_block, total_size)

    CUDA.@time @cuda threads=threads_actual blocks=Int(ceil(total_size / threads_actual)) computePixelKernel!(image, quadrants, brightnesses, result)
    return colorview(RGB, Array(result))
end
