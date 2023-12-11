train_lines = []
val_lines   = []

train_file  = 'train.txt'
val_file    = 'val.txt'
total_data_num = 34149

# Define the split point
split_point = 28130

for i in range(total_data_num):
    i_string = "%06d\n" % i
    if i < split_point:
        train_lines.append(i_string)
    else:
        val_lines.append(i_string)

with open(train_file, 'w') as file:
    file.writelines(train_lines)

with open(val_file, 'w') as file:
    file.writelines(val_lines)

