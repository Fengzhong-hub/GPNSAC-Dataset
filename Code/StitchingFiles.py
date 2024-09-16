import os

# List of txt files to be concatenated
file_list = ['/home/wanghangyu/pythonProjects/testProject/data/data1.log',
             '/home/wanghangyu/pythonProjects/testProject/data/data2.log',
             '/home/wanghangyu/pythonProjects/testProject/data/data_tamper.log',
             '/home/wanghangyu/pythonProjects/testProject/data/data_steal.log',
             '/home/wanghangyu/pythonProjects/testProject/data/data_dos.log']  # Replace with your file name

# Output File
output_file = '/home/wanghangyu/pythonProjects/testProject/data/output.log'

# Open output file for writing
with open(output_file, 'w', encoding='utf-8') as outfile:
    for file_name in file_list:
        # Check if a file exists
        if os.path.exists(file_name):
            # Open each file in turn and read the contents
            with open(file_name, 'r', encoding='utf-8') as infile:
                # Read all lines
                lines = infile.readlines()

                # If the last line is blank, remove it
                if lines and lines[-1].strip() == '':
                    lines = lines[:-1]

                # Write the contents to the output file
                outfile.writelines(lines)

        else:
            print(f"{file_name} is not exist.")
