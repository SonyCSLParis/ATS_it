import os

# giving directory name
dir_Teacher = '/Users/francesca/Desktop/CORPORA_TEXT_SIMP/Teacher'
dir_Terence = '/Users/francesca/Desktop/CORPORA_TEXT_SIMP/Terence'
# giving file extension
ext = 'txt'
file_to_parse_1 = []
file_to_parse_2 = []
# iterating over all files

for file in os.listdir(dir_Teacher):
    if file.endswith(ext):
        final_dir = dir_Teacher + '/' + file
        file_to_parse_1.append(final_dir)

print(len(file_to_parse_1))

for file in os.listdir(dir_Teacher):
    if file.endswith(ext):
        final_dir = dir_Teacher + '/' + file
        file_to_parse_1.append(final_dir)
'''with open(file, 'r') as infile:  # printing file name of desired extension
    for line in infile.readlines():
        print(line)

        continue'''