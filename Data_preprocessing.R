# Splitting the dataset into train, test and validation

# Set the path to your dataset directory
data_dir <- "./dataset"

# Set the path to store the divided data
output_dir <- "./weather"

# Create the output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Get a list of all class folders in the dataset directory
class_folders <- list.dirs(data_dir, recursive = FALSE)

# Function to copy files to the destination directory
copy_files <- function(src_files, dest_dir) {
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(src_files, dest_dir)
}

# Divide the data for each class into 80% train, 10% test, and 10% validation
for (class_folder in class_folders) {
  class_name <- basename(class_folder)
  output_class_dir <- file.path(output_dir, "train", class_name)
  train_files <- list.files(class_folder, full.names = TRUE, recursive = FALSE)
  train_files <- sample(train_files, floor(0.8 * length(train_files)))
  copy_files(train_files, output_class_dir)
  
  remaining_files <- setdiff(list.files(class_folder, full.names = TRUE, recursive = FALSE), train_files)
  output_class_dir <- file.path(output_dir, "test", class_name)
  test_files <- sample(remaining_files, floor(0.5 * length(remaining_files)))
  copy_files(test_files, output_class_dir)
  
  validation_files <- setdiff(remaining_files, test_files)
  output_class_dir <- file.path(output_dir, "validation", class_name)
  copy_files(validation_files, output_class_dir)
}

