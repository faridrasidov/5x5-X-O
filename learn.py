import os.path as path

script_path = path.dirname(path.abspath(__file__))
learning_data = script_path + '/learn_data.txt'
learning_output = script_path + '/learn_output.txt'
corrupt_data = script_path + '/corrupted_data.txt'
lost_data = script_path + '/lost_data.txt'

def load_learn_data(file_name: str = None):
    """
    loads leaning datas
    """

    def separate_arr(array, where):
        """
        func to separate X & O arrays to put on main_dict
        """
        main_temp_value = main_dict[where]
        main_temp_value.append(array)
        main_dict[where] = main_temp_value

    main_dict = {
        '1': [],
        '-1': []
    }

    if not file_name:
        file_name = learning_data

    with open(file_name, 'r', encoding='utf-8') as learn_file:
        for line in learn_file.readlines():
            line_nums_str = line.split(' ')
            temp_arr = []
            for str_num in line_nums_str[:-1]:
                temp_arr.append(int(str_num))

            if int(line_nums_str[-1]) == -1:  # if its O
                separate_arr(temp_arr, '-1')
            else:  # if its X
                separate_arr(temp_arr, '1')

    return main_dict


def heb_func(all_data):
    """
    func calculate the correct weight & bias value and return them
    """
    weight_vector = [0] * 25
    bias = 0
    for item in all_data:
        for input_vector in all_data[item]:
            vector_len = len(input_vector)
            for i in range(vector_len):
                weight_vector[i] = weight_vector[i] + (input_vector[i] * int(item))
            bias = bias + (1 * int(item))
    return weight_vector, bias


if __name__ == '__main__':
    learn_dict = load_learn_data()
    learn_weight, learn_bias = heb_func(learn_dict)
    with open(learning_output, 'w', encoding='utf-8') as output:
        output.write(f"{learn_weight}\n{learn_bias}")
    print(f'learning datas saved.')
