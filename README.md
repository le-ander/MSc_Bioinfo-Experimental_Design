# Experimental-Design

14/2/2017
To do list:

# Important

- Entropy calculations
  - Include launch configuration in getEntropy functions
  - Rewrite getEntropy3

- Combine all parts together

- Figure out Python 2.7 error

# Next steps

- Check why we get such large numbers in res1 and res_t2

- Stochastic implementation
  - Reuse code from MEANS package
  - Adapt CUDAsim SDE solver to handle multiple models

- Error-checking
  - Catch user input errors
  - Only load packages when required (libsbml and means)
  - To be done at the end

- Comment code extensively

- Presentation of results
  - Terminal output
    - Log file
  - Graphs

- Group report
  - Split work (assign sections)
  - Write it

# Extra

- Update abcsysbio parser to take local parameters

- GPU optimisations
  - Optimize grid shape and adaptive grid size
  - Optimize Block shape and Grid shape
  - Move first summations to GPU
  - Reduce bottleneck of shuttling files between device and host by reducing grid size and copying data for n runs onto GPU
  - Add third dimension to blocks/grid to improve performance, use smem?

- getEntropy optimisations
  - Convert part B of getEntropy2 to 2 dimensions

- Improve CUDA-sim implementation
  - Scott's for-loop to be pushed to new update
  - Fix CUDAsim device detection
  - Using launch configuration in CUDAsim to optimise block size and grid size

- Convert to Python 3.5
  - Modernize package
  - Use Six package for compatibility

- Julia wrapper
