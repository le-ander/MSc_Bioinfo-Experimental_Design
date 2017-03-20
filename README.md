# Experimental-Design ToDo

# Next steps

- Error-checking
  - Try breaking code (Jonas)
  - Catch div by 0 in gE due to all inf/na (Leander)

- Finish commenting (incl. kernel, incl referencing) (Scott, Jonas, Leander)

- Rename (models, functions, folder struct, (vars)) (?)

- Presentation of results
  - Finalise Print statements (Scott, Jonas, Leander)
  - CSV summary file (append not overwrite) (?)
  - Bar graphs (Jonas)

- Usability
  - Pickling (reuse cudasim)  (Scott)
  - Only load packages when required (libsbml and means) (?)

- Final run through code (rm print statements, rm seeds) (Scott, Jonas, Leander)

# On Hold

- Increase gridsize as less global mem is used now (Leander)

- Dynamically assign shared memory base on block size (Leander, Scott)

- SDE implementation (Scott, Leander)

- HDF5 to avoid memory issues (Jonas)

- Use analytical model to validate project (see Julianes thesis) (?)

# Further work ideas

- Improve CUDA-sim implementation
  - Scott's for-loop to be pushed to new update
  - Fix CUDAsim device detection

- Convert to Python 3.5
  - Modernize package
  - Use Six package for compatibility
