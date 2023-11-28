#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

# Input file containing URLs
input_file="$1"

# Output file to concatenate the downloaded .txt files
output_file="$2"

# Create a temporary directory to store downloaded files
temp_dir=$(mktemp -d)

# Function to download a file from a URL
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    local output_path="$temp_dir/$filename"

    curl -sS "$url" -o "$output_path"
}

# Iterate over each URL in the input file
while IFS= read -r url; do
    echo "Downloading $url"
    download_file "$url"
done < "$input_file"

# Concatenate all downloaded .txt files into a single file
cat "$temp_dir"/*.txt > "$output_file"

# Clean up temporary directory
rm -r "$temp_dir"

echo "Download and concatenation completed. Output saved to $output_file"

# Print the number of lines in the output file
echo "Number of lines in output file: $(wc -l < "$output_file")"
