import os


def file_name(file_dir):
    file_list = []
    dirs_list = []
    file_name_list =[]
    for root, dirs, files in os.walk(file_dir):
        dirs_list.append(dirs)
        for file in files:
            file_list.append(file)
            file_name_list.append(os.path.join(root,file))
    return file_list, file_name_list, dirs_list

