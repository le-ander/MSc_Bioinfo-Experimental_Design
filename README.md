# Experimental-Design ToDo

# Important


- gE3:
  - Write run_gE3
  - Work on scaling_gE3
    - Ensure that different number of timepoints can exist for each model in gE3

- Combine all parts together and test


# Next steps

- Make sure cudaSim works wit gE3 (Scott)

- Reduce number of simulations before gE3 (gE1&2 as well?) and slice arrays more efficiently
  - dataRef/Mod (N1); thetaRef (N2+N3); thetaMod (N2+N4)

- Check why we get such large numbers in res1 and result

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
