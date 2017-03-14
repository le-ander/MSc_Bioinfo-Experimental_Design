# Experimental-Design ToDo

# Next steps

- Check why we get such large numbers in res1 and result

- Sort out scaling

- Stochastic implementation
  - Reuse code from MEANS package
  - Adapt CUDAsim SDE solver to handle multiple models

- Error-checking
  - Catch user input errors
  - Only load packages when required (libsbml and means)
  - Catch div by 0 in gE due to all inf/na

- Comment code extensively

- Tackle memory issues of gE3 (split up computations)

- Presentation of results
  - Terminal output
    - Log file
  - Graphs

- Group report
  - Split work (assign sections)
  - Write it

- REMOVE SEEDS

# Extra

- Dynamically plotting trajectories on graph

- Update abcsysbio parser to take local parameters

- GPU optimisations
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
