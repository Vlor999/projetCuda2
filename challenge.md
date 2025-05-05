# Challenge

## Progress Overview

### First Optimization
- Utilized low shared memory.
- Implemented multiple threads per pixel (warp-sized).
- Applied multithreading across the entire image.

### Second Optimization (Unsuccessful)
- Experimented with larger block sizes to assess potential improvements.
- Attempted to use CUB library, but faced challenges:
    - Compatibility issues with my system.
    - Possible lack of proper understanding of its usage.

### Third Optimization
- Consolidated prefix sum, weight calculations, and final color determination into a single kernel launch.
- Leveraged warp-level and multithreading techniques for all pixels.
- Experimented with binary operations (`<<`, `>>`, etc.) to minimize unnecessary cycles (found to be ineffective).

### Conclcusion
I started, on my computer, with some GPU computations, to have some times that were around 600ms for the `data/lego_front_256` and fish with some times that are around 90ms per iterartions.