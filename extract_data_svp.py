import os
import shutil

train_dir = "ImageNet1000/imagenet-mini/train"
output_train_dir = "bananas/train"
img_height = 224
img_width = 224

def gen_image_data(path, files, height, width):
  image_data = []
  t = 0
  for f in files:
    dir = path + "/"
    image_string = tf.io.read_file(dir + f)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [height, width])
    t+=1

    if image_resized.shape[2] == 1:
      image_resized = tf.image.grayscale_to_rgb(image_resized)

    image_data.append(image_resized)

  return tf.convert_to_tensor(image_data)

def gen_one_hot_labels(dirs, map, num_classes=1000):
  labels = np.zeros((map["_stats"]["count"], num_classes))
  start_index = 0
  end_index = 0

  DEBUG_BREAK = 0 # remove
  for i, dir in enumerate(dirs):
    end_index += map[dir]["count"]
    labels[start_index:end_index, i] = 1.0
    start_index += map[dir]["count"]

    DEBUG_BREAK += 1 # remove
    if DEBUG_BREAK == 5: # remove
      break # remove

  return tf.convert_to_tensor(labels)

def list_dir(directory):
  dirs = os.listdir(directory)
  classes = {dir: {key: None for key in ["class_num", "count"]} for dir in dirs}
  classes["_stats"] = {key: None for key in ["min", "max", "avg", "median", "count"]}
  files = {"full_path": [], "filename": [], "class": []}
  files_count = []

  base_dir = directory + "/"
  DEBUG_BREAK = 0 # remove
  for i, dir in enumerate(dirs):
    file_list = os.listdir(base_dir + dir)
    sub_dir = dir + "/"
    files["filename"] += file_list
    file_list = [sub_dir + f for f in file_list]
    files["full_path"] += file_list
    files["class"].append(dir)
    classes[dir]["count"] = len(file_list)
    classes[dir]["class_num"] = i
    files_count.append(len(file_list))

    DEBUG_BREAK += 1 # remove
    if DEBUG_BREAK == 5: # remove
      break # remove

  classes["_stats"]["min"] = np.min(files_count)
  classes["_stats"]["max"] = np.max(files_count)
  classes["_stats"]["avg"] = np.mean(files_count)
  classes["_stats"]["median"] = np.median(files_count)
  classes["_stats"]["count"] = np.sum(files_count)

  return dirs, files, classes

def create_new_dataset(files, indices, dirs, input_path, output_path, force_overwrite=False):
  path_split = output_path.split("/")
  assert os.path.exists(input_path), "input path non-existing"

  if output_path[len(output_path) - 1] == "/":
    path_split = path_split[:-1]

  if os.path.exists(output_path):
    if force_overwrite:
      shutil.rmtree(path_split[0])
      os.makedirs(output_path)
    else:
      assert len(os.listdir(output_path)) == 0, "output path exists and is not empty"
  else:
    try:
      os.makedirs(output_path)
    except Exception:
      print("Failed creating output dir")

  try:
    for i, ind in enumerate(indices):
      dir, filename = files["full_path"][ind].split("/")
      dir_full = output_path + dir if output_path[len(output_path) - 1] == "/" else output_path + "/" + dir

      if not os.path.exists(dir_full):
        os.mkdir(dir_full)

      source_path = input_path + dir + "/" + filename if input_path[len(input_path) - 1] == "/" else input_path + "/" + dir + "/" + filename
      shutil.copy(source_path, dir_full)
  except Exception as e:
    print(e)
    print("Failed to copy " + dir + "/" + filename)


sorted_dirs, files, class_map = list_dir(train_dir)
train_x = gen_image_data(train_dir, files["full_path"], img_height, img_width)
train_y = gen_one_hot_labels(sorted_dirs, class_map)

###
# DO SVP STUFF!
# REMEMBER: DOES IT ALTER SORTING OF TRAINING DATA IN-PLACE?

entropy_index = [4, 56, 21, 106, 199, 3] # RANDOM TOY INDICES

create_new_dataset(files, entropy_index, sorted_dirs, train_dir, output_train_dir)