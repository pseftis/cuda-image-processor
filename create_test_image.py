#!/usr/bin/env python3
"""
Helper script to create a test PPM image for testing the CUDA image processor.
This creates a simple pattern image that can be used to verify the blur effect.
"""

import struct
import sys

def create_test_ppm(filename, width=1024, height=1024):
    """Create a test PPM image with a checkerboard pattern."""
    with open(filename, 'wb') as f:
        # Write PPM header
        header = f"P6\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        
        # Write pixel data (checkerboard pattern)
        for y in range(height):
            for x in range(width):
                # Create a colorful checkerboard pattern
                check = ((x // 64) + (y // 64)) % 2
                if check:
                    # Bright colors
                    r = (x * 255) // width
                    g = (y * 255) // height
                    b = 128
                else:
                    # Dark colors
                    r = 50
                    g = 50
                    b = 50
                
                f.write(struct.pack('BBB', r, g, b))
    
    print(f"Created test image: {filename} ({width}x{height})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        width = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        height = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    else:
        filename = "test_input.ppm"
        width = 1024
        height = 1024
    
    create_test_ppm(filename, width, height)

