# Experimental-Design ToDo

## Ask Juliane

- How to present repressilator results?
- Should libsbml be a dependency or not?

## Next steps

- Error-checking (try breaking code) (Jonas)

- Finish commenting (incl. kernel, incl referencing) (Scott, Jonas, Leander)

- Rename (models, functions, vars) (?)

- Presentation of results
  - Finalise Print statements (Scott, Jonas, Leander)
  - CSV summary file (append not overwrite) (?)
  - Bar graphs (Jonas)

- Only load packages when required (libsbml and means) (?)

- At very end: Final run through code (rm print statements, rm seeds) (Scott, Jonas, Leander)

- Peitho installation (Jonas)

- Run timing runs

- Group report
  - Include figure for block/grid details?
  - Figure out how to present repressilator results
  - Add results for examples
  - Write Troubleshooting
  - Write Future Work
  - Complete author contributions (Chapter 1 and appendix A)

- Read through group report

## On Hold

- Increase gridsize as less global mem is used now (Leander)

- Dynamically assign shared memory based on block size (Scott)

- Include dynamic shared memory in launch config (consider max smem per SM) (Leander)

- SDE implementation (Scott, Leander)

- HDF5 to avoid memory issues (Jonas)

- Pickling (reuse cudasim)  (Jonas)

- Use analytical model to validate project (see Julianes thesis) (?)

## Further work ideas

- Improve CUDA-sim implementation
  - Scott's for-loop to be pushed to new update
  - Fix CUDAsim device detection

- Convert to Python 3.5
  - Modernize package
  - Use Six package for compatibility
