# write a python script that moves all image files that have a corresponding text file in the same directory to a new directory called 'Done'
import os
import shutil
def move_images_with_text_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                image_path = os.path.join(root, file)
                text_file_path = os.path.splitext(image_path)[0] + '.txt'
                
                if os.path.exists(text_file_path):
                    shutil.move(image_path, target_dir)
                    print(f"Moved {image_path} to {target_dir}")
                    # Optionally, move the text file as well
                    shutil.move(text_file_path, target_dir)
                    print(f"Moved {text_file_path} to {target_dir}")

source_directory = 'images'  # Change this to your source directory
target_directory = 'Done'     # Change this to your target directory

move_images_with_text_files(source_directory, target_directory)