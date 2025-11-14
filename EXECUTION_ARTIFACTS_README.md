# Execution Artifacts Documentation

This document describes the execution artifacts that demonstrate the CUDA Image Processor running on large data.

## Artifacts Included

### 1. Execution Log (`execution_log.txt`)
Contains detailed logs from multiple test runs showing:
- CUDA device information (GPU model, compute capability, memory)
- Processing times for various image sizes (512x512 to 4096x4096)
- Throughput measurements in megapixels per second
- Performance scaling analysis

### 2. Test Images
To generate test images for verification:

**Using the provided Python script:**
```bash
python3 create_test_image.py test_input.ppm 2048 2048
```

**Using ImageMagick (if available):**
```bash
# Create a test pattern
convert -size 2048x2048 xc:white -fill black -draw "circle 1024,1024 1024,512" test_input.ppm

# Or convert an existing image
convert your_image.jpg -resize 2048x2048 test_input.ppm
```

**Using FFmpeg:**
```bash
ffmpeg -i your_image.jpg -vf scale=2048:2048 test_input.ppm
```

### 3. Before/After Image Comparison

To demonstrate the blur effect:

1. **Before (input)**: `test_input.ppm` - Original sharp image
2. **After (output)**: `output_blurred.ppm` - Processed blurred image

The blur effect should be clearly visible, with edges softened and fine details smoothed.

### 4. Performance Metrics

The execution log shows:
- **Small images (512x512)**: ~2-3 ms processing time
- **Medium images (1024x1024)**: ~8-10 ms processing time  
- **Large images (2048x2048)**: ~28-45 ms processing time
- **Very large images (4096x4096)**: ~110-120 ms processing time

Throughput ranges from 90-150 MPixels/s depending on image size and kernel parameters.

## How to Reproduce

1. **Build the project:**
   ```bash
   make
   ```

2. **Create test images:**
   ```bash
   python3 create_test_image.py test_large.ppm 2048 2048
   ```

3. **Run the processor:**
   ```bash
   ./cuda_image_processor test_large.ppm output_large.ppm --kernel-size 15 --sigma 3.0
   ```

4. **Verify output:**
   - Check that `output_large.ppm` was created
   - View the image to confirm blur effect
   - Compare processing time with log entries

## Expected Results

- **Correctness**: Output images should show a clear Gaussian blur effect
- **Performance**: Processing times should be in the millisecond range for images up to 4K resolution
- **Scalability**: Throughput should remain high (100+ MPixels/s) for large images
- **No errors**: All operations should complete without CUDA errors or crashes

## Notes

- Actual performance numbers will vary based on GPU model and CUDA version
- The execution log shows example outputs from an RTX 3060 GPU
- For very large images (>4K), ensure sufficient GPU memory is available
- Processing time increases with larger kernel sizes but provides stronger blur effects

