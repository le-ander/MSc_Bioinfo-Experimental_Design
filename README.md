# Experimental-Design

10/2/2017
To do list:

# Important
- Parsing
  - Implement changes from today's meeting
  - Combine all elements of parsing
  - Local code and input.xml files
  - Add N (N1&N2) to model object at start
  
- Simulation and Data Processing
  - Update functions to be used with parsing output
  - Jonas to add attributes to model object
  
- Entropy calculations
  - Include launch configuration in getEntropy functions
  - Rewrite getEntropy2 in Repressilator and getEntropy 1.5 in Hes1 folder

- Combine all parts together

# Next steps
- Time original code

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
  
- GPU optimisations
  - Optimize grid shape and adaptive grid size
  - Optimize Block shape and Grid shape
  - Move first summations to GPU
  - Reduce bottleneck of shuttling files between device and host by reducing grid size and copying data for n runs onto GPU
  - Add third dimension to blocks/grid to improve performance, use smem?

- Improve CUDA-sim implementation
  - Scott's for-loop to be pushed to new update
  - Fix CUDAsim device detection
  - Using launch configuration in CUDAsim to optimise block size and grid size
  
- Convert to Python 3.5
  - Modernize package
  - Use Six package for compatibility

- Julia wrapper
