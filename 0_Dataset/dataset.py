# %%
# Installing gdown
!pip install -U --no-cache-dir gdown --pre

# Drive Link
# https://drive.google.com/file/d/1JqeVC7BJllTTsVAQv1KS2uIFS7Iq_Upf/view?usp=sharing

# Downloaading dataset
!gdown 1JqeVC7BJllTTsVAQv1KS2uIFS7Iq_Upf

# %%
!rm -rf sample_data
!unzip archive.zip
!rm archive.zip

# %%
!sudo pip install ipdb


# %%
# let us clean
%cd /content/
!mv /content/annotations/annotations_YOLO /content/
!rm -rf annotations
!rm *.txt
!mv annotations_YOLO annots
!mv /content/scripts/frameExtractor.py /content/
!rm -rf scripts

# %%
import os
# filename = "2019-02-22_22-31-28to2019-02-22_22-31-38_1"
# os.system('ffmpeg -i /content/dataset/videos/crab/{}.avi -vf scale=960:540 -sws_flags bicubic {}-%04d.png -hide_banner'.format(filename,filename))


# %%
!mkdir images


# %%
!mkdir videos
!mv /content/dataset/videos/*/* /content/videos

# %%
!rm -rf dataset

# %%
%cd images
import os
for eachfile in os.listdir("/content/videos"):
  filename = eachfile[:-4]
  os.system('ffmpeg -i /content/videos/{}.avi -vf scale=960:540 -sws_flags bicubic {}-%04d.jpg -hide_banner'.format(filename,filename))

# %%
%cd /content/
!rm -rf videos

# %%
!rm frameExtractor.py

# %%
# check whether all files coexist in each folder

import os

# Set the paths to the two folders
folder1 = "annots"
folder2 = "images"

# Get the list of files in each folder
files1 = [os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
files2 = [os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

# Compare the lists of file names ignoring extensions
common_files = set([f1 for f1 in files1 for f2 in files2 if f1 == f2])
if len(common_files) > 0:
    print("The following files exist in both folders")
    for f in common_files:
        # print(f)
        continue
else:
    print("No common files found.")


# %%
import os

# Set the paths to the two folders
images_folder = "images"
annotations_folder = "annots"

# Get the list of files in each folder
image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
annotation_files = [f for f in os.listdir(annotations_folder) if os.path.isfile(os.path.join(annotations_folder, f))]

# Sort the file lists by name to ensure consistent ordering
image_files.sort()
annotation_files.sort()

# Rename the files in both folders
for i, (image_file, annotation_file) in enumerate(zip(image_files, annotation_files), start=1):
    # Create new file names with a numeric sequence
    new_image_file = f"{i}.png"
    new_annotation_file = f"{i}.txt"

    # Rename the files in the 'images' folder
    os.rename(os.path.join(images_folder, image_file), os.path.join(images_folder, new_image_file))

    # Rename the files in the 'annotations' folder
    os.rename(os.path.join(annotations_folder, annotation_file), os.path.join(annotations_folder, new_annotation_file))

    # Print the original and new file names for each file
    print(f"Renamed '{image_file}' to '{new_image_file}' in the 'images' folder.")
    print(f"Renamed '{annotation_file}' to '{new_annotation_file}' in the 'annotations' folder.")


# %%
# check whether all files coexist in each folder

import os

# Set the paths to the two folders
folder1 = "annots"
folder2 = "images"

# Get the list of files in each folder
files1 = [os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
files2 = [os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

# Compare the lists of file names ignoring extensions
common_files = set([f1 for f1 in files1 for f2 in files2 if f1 == f2])
if len(common_files) > 0:
    print("The following files exist in both folders:")
    for f in common_files:
        # print(f)
        continue
else:
    print("No common files found.")


# %%


# %%
# !mv annots labels
import os
os.rename("annots", "labels")

# %%
%cd images

# %%
!sudo apt-get install imagemagick

# %%
!mogrify

# %%
!pwd

# %%
%cd /content/
os.rename("images", "pngs")
!mkdir images
%cd pngs


# %%
!mogrify -path /content/images -format jpg *.png

# %%
%cd /content/

# %%
!ls

# %%
!rm -rf pngs

# %%


# %%
!ls

# %%
!ls /content/labels | wc -l

# %%
!ls /content/images | wc -l

# %%
# LETS REMOVE THE DATASET THAT DO NOT HVAE LABELS and information

# %%
import os

image_dir = "images/"
label_dir = "labels/"

# get a list of all the YOLO label files in the label directory
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

# loop through the label files
for label_file in label_files:
    # check if the label file is empty
    if os.stat(label_dir + label_file).st_size == 0:
        # if it's empty, remove the label file
        os.remove(label_dir + label_file)
        
        # also remove the corresponding image file
        image_file = label_file.split(".")[0] + ".jpg"
        os.remove(image_dir + image_file)
        print("removed successfully: ",image_file)


# %%
!ls /content/labels | wc -l

# %%
!ls /content/images | wc -l

# %%
# Normalise the coordinates

# %%
def normalize_yolo_coords(yolo_txt_file_path):
    image_width, image_height = 960, 540
    # Read the YOLO text file
    with open(yolo_txt_file_path, 'r') as f:
        lines = f.readlines()
    
    # Normalize the coordinates of each line
    for i, line in enumerate(lines):
        # Split the line into parts
        parts = line.split()
        # Extract the class label and the coordinates
        class_label = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Normalize the coordinates
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        # # Correcting class labels
        # if(class_label==1):
        #   class_label=0
        # elif(class_label==2):
        #   class_label=1
        # elif(class_label==3):
        #   class_label=2
        # elif(class_label==4):
        #   class_label=3
        # elif(class_label==5):
        #   class_label=4
        # elif(class_label==6):
        #   class_label=5

        # Update the line with the normalized coordinates
        lines[i] = f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    
    # Write the normalized coordinates back to the YOLO text file
    with open(yolo_txt_file_path, 'w') as f:
        f.writelines(lines)


# %%
!cp -r labels labels_backup

# %%
!pwd

# %%
for eachfile in os.listdir("/content/labels/"):
  path = "/content/labels/"+eachfile
  normalize_yolo_coords(path)

# %%
!rm -rf labels_backup

# %%
!ls

# %%
!zip -r /content/file.zip /content/

# %%
os.rename("file.zip","dataset.zip")

# %%


# %%


# %%



