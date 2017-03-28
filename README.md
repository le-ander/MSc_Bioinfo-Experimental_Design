# Experimental-Design ToDo

## Ask Juliane

- How to present repressilator results?
- Should libsbml be a dependency or not?
- Should we have a table with author contributions at the end of the report?

## Next steps

- Finish commenting (incl. kernel, incl referencing) (Scott, Jonas, Leander)

- Rename (models, functions, vars, files ) (?)

- Presentation of results
  - Finalise Print statements (Scott, Jonas, Leander)
  - CSV summary file (append not overwrite) (?)
  - Bar graphs (Jonas)

- Peitho installation (Jonas)

- Run timing runs

- Group report
  - Figure out how to present repressilator results (incl. results)
  - Complete Troubleshooting (Jonas)
  - Write Future Work

- Read through group report

- At very end: Final run through code (rm print statements, rm seeds) (Scott, Jonas, Leander)

## On Hold

- Increase gridsize as less global mem is used now (Leander)

- Include dynamic shared memory in launch config (consider max smem per SM) (Leander)

- Dynamically assign shared memory based on block size (Scott)

- SDE implementation (Scott, Leander)

- HDF5 to avoid memory issues (Jonas)

- Pickling (reuse cudasim)  (Jonas)

- Only load packages when required (libsbml and means) (wati for Julianes opinion)

## Further work ideas

- Improve CUDA-sim implementation
  - Scott's for-loop to be pushed to new update
  - Fix CUDAsim device detection

- Convert to Python 3.5
  - Modernize package
  - Use Six package for compatibility
