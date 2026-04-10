#!/bin/bash

# Enforce strict error handling
set -e

# Validate the number of provided arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <ur_type> <hand_type> <flange_type>"
    echo "Example: $0 ur10e inspire_strong_hand ring"
    exit 1
fi

# Assign arguments to explicitly named variables
UR_TYPE=$1
HAND_TYPE=$2
FLANGE_TYPE=$3

# Determine the filename suffix based on the flange_type parameter
FLANGE_SUFFIX=""
if [ "${FLANGE_TYPE}" != "none" ]; then
    FLANGE_SUFFIX="_flange"
fi

# Define strict relative paths
INPUT_FILE="src/moveit_ur_hand_config/config/workcell.urdf.xacro"
OUTPUT_DIR="src/moveit_ur_hand_config/urdf"
OUTPUT_FILE="${OUTPUT_DIR}/${HAND_TYPE}_${UR_TYPE}${FLANGE_SUFFIX}_workcell.urdf"

# Verify the input file exists before attempting compilation
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file ${INPUT_FILE} does not exist."
    exit 1
fi

# Ensure the output directory structure exists
mkdir -p "${OUTPUT_DIR}"

# Execute the compilation and redirect stdout to the dynamically named file
echo "Compiling Xacro configuration..."
xacro "${INPUT_FILE}" ur_type:="${UR_TYPE}" hand_type:="${HAND_TYPE}" flange_type:="${FLANGE_TYPE}" > "${OUTPUT_FILE}"

# Confirm successful execution
echo "Success: URDF generated at ${OUTPUT_FILE}"