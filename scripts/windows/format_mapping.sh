#!/bin/bash
#
# Run script from top of project dir
FP=nsdata/mappings
for file in "$FP"/*.json ; do
    fixjson $file > $file
done
