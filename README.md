# Experimental-Design

14/2/2017
To do list:

# Important
- Parsing
  - Combine all elements of parsing

- Simulation and Data Processing
  - Update functions to be used with parsing output
  
- Entropy calculations
  - Include launch configuration in getEntropy functions
  - Rewrite getEntropy2 in Repressilator and getEntropy 1.5 in Hes1 folder

- Combine all parts together

# Next steps

- Prepare presentation

- Check that scaling function still gives same results as before

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
