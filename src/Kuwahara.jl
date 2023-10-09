function kuwahara(image_in, region_size::Int=13)
    image = RGB.(image_in)
    brightnesses = channelview(float.(HSV.(image)))[3, :, :]
    result = similar(image)

    quadrant_size = Int(ceil(region_size / 2))

    avgColor(colors) = RGB(
        sum(map(c -> c.r, colors), init=0) / length(colors),
        sum(map(c -> c.g, colors), init=0) / length(colors),
        sum(map(c -> c.b, colors), init=0) / length(colors)
    )

    keepInBounds(arr, dim) = filter(coord -> coord > 0 && coord <= size(image, dim), arr)

    function computePixel(y, x)
        top = y - (region_size ÷ 2)
        left = x - (region_size ÷ 2)

        quad_1 = [keepInBounds(top:(top+quadrant_size-1), 1), keepInBounds(left:(left+quadrant_size-1), 2)]
        quad_2 = [keepInBounds(top:top+quadrant_size-1, 1), keepInBounds((left+quadrant_size-1):(left+region_size-1), 2)]
        quad_3 = [keepInBounds((top+quadrant_size-1):(top+region_size-1), 1), keepInBounds(left:(left+quadrant_size-1), 2)]
        quad_4 = [keepInBounds((top+quadrant_size-1):(top+region_size-1), 1), keepInBounds((left+quadrant_size-1):(left+region_size-1), 2)]
        
        std_1 = std(brightnesses[quad_1...])
        std_2 = std(brightnesses[quad_2...])
        std_3 = std(brightnesses[quad_3...])
        std_4 = std(brightnesses[quad_4...])

        quadrant = [quad_1, quad_2, quad_3, quad_4][argmin([std_1, std_2, std_3, std_4])]

        result[y, x] = avgColor(image[quadrant...])
    end

    @time "computation" Threads.@threads for y in 1:size(image, 1)
        for x in 1:size(image, 2)
            computePixel(y, x)
        end
    end
    return result
end
