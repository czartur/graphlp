#!/bin/bash
PYTHON=python3
OUTPUT=results.csv

extract_error() {
    error_line=$($PYTHON main.py --load-path="$1" --no-plot| grep "mean distance error")
    echo "${error_line##*= }"
}

print_help() {
    echo "Usage: ./script.sh [OPTIONS]"
    echo "Options:"
    echo "  --help, -h       Display this help menu"
    echo "  --folder-path, -f  Specify the path to the folder containing models to benchmark"
    exit 0
}

if [ $# -eq 0 ]; then
    echo "No options selected"
    print_help
fi

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_help
            ;;
        -f|--folder-path)
            folder_path="$2"
            shift
            ;;
        *)
            echo "Invalid option: $key"
            print_help
            ;;
    esac
    shift
done

if [ ! -d "$folder_path" ]; then
    echo "Error: '$folder_path' does not exist or is not a directory"
    exit 1
fi


echo "File,Value" > $OUTPUT
for file in "$folder_path"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        value=$(extract_error "$file")
        echo "$filename,$value" >> $OUTPUT
        echo "$filename: MDE = $value"
    fi
done

if [ $? -eq 0 ]; then
  echo "Results saved to '$OUTPUT'"
fi

