# Experimental-Design ToDo

# Next steps

- Error-checking
  - Catch user input errors
  - Only load packages when required (libsbml and means)
  - Catch div by 0 in gE due to all inf/na

- Comment code

- Optimise getEntropy function for speed
  - Optimise general code structure
  - Reduction kernel on GPU (reducing data to be move back onto host)

- Stochastic implementation
  - Reuse code from MEANS package
  - Adapt CUDAsim SDE solver to handle multiple models

- Presentation of results
  - Terminal output
  - Log file
  - Graphs

- Write group report

- Write individual reports

- Remove Seeds

# On Hold

- Sort out scaling (check why we get such large numbers in res1 and result)

- Tackle memory issues of gE3 (split up computations)

- Update abcsysbio parser to take local parameters

- getEntropy optimisations
  - Convert part B of getEntropy2 to 2 dimensions

# Further work ideas

- Dynamically plotting trajectories on a graph

- GPU optimisations
  - Reduce bottleneck of shuttling files between device and host by reducing grid size and copying data for n runs onto GPU
  - Add third dimension to blocks/grid to improve performance, use smem?

- Improve CUDA-sim implementation
  - Scott's for-loop to be pushed to new update
  - Fix CUDAsim device detection

- Convert to Python 3.5
  - Modernize package
  - Use Six package for compatibility

- Julia wrapper
