#!/bin/bash

# Check if earthengine command-line tool is installed
if ! command -v earthengine &> /dev/null
then
    echo "earthengine command-line tool not found. Please install it first."
    exit 1
fi

# Variables: Image Collection path and maximum parallel processes
COLLECTION_PATH="$1"
MAX_PARALLEL=10  # Maximum number of parallel processes

# Check for input arguments
if [ -z "$COLLECTION_PATH" ]; then
  echo "Usage: $0 <collection_path>"
  exit 1
fi

# Get the list of assets in the collection
ASSETS=$(earthengine ls "$COLLECTION_PATH")
ASSET_COUNT=$(echo "$ASSETS" | wc -l)

# If no assets found, exit
if [ -z "$ASSETS" ]; then
  echo "No assets found in the collection."
  exit 0
fi

# Prompt user for confirmation
echo "Found $ASSET_COUNT assets in $COLLECTION_PATH."
read -p "Do you want to delete these assets from $COLLECTION_PATH? (y/n): " CONFIRMATION

# If user does not confirm, exit
if [[ "$CONFIRMATION" != "y" ]]; then
  echo "Operation canceled."
  exit 0
fi

# Function to delete an asset
delete_asset() {
  local asset="$1"
  echo "Deleting $asset"
  earthengine rm "$asset"
}

# Delete assets in parallel, limiting to $MAX_PARALLEL parallel processes
echo "Deleting $ASSET_COUNT assets from $COLLECTION_PATH with max $MAX_PARALLEL in parallel"

for asset in $ASSETS; do
  # Start the delete in the background
  delete_asset "$asset" &

  # Limit the number of parallel processes
  while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 0.05  # Wait until there's a free slot
  done
done

# Wait for all background jobs to finish
wait

# Check for optional flag to delete the collection itself
DELETE_COLLECTION=false
if [[ "$2" == "--delete-collection" ]]; then
  DELETE_COLLECTION=true
fi

# Optionally delete the collection after all assets are removed
if [ "$DELETE_COLLECTION" = true ]; then
  echo "Deleting the collection: $COLLECTION_PATH"
  earthengine rm "$COLLECTION_PATH"
fi

echo "Image Collection deletion process completed."
