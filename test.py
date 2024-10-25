import ast

import numpy as np

from learn import load_learn_data, corrupt_data, learning_output, lost_data


def classify(input_pattern, w_vector, b):
    """
    classifies the input pattern
    """
    result = np.dot(input_pattern, w_vector) + b
    return 1 if result >= 0 else -1


def test_heb_network(all_dict, dict_name):
    """
    test the Heb network with loaded datas.
    some_d: Dictionary of input patterns for 'X' and 'O'.
    """
    correct_X, correct_O = 0, 0

    # Test 'X' patterns
    for pattern in all_dict['1']:
        prediction = classify(pattern, weight_vector, bias)
        if prediction == 1:
            correct_X += 1

    # Test 'O' patterns
    for pattern in all_dict['-1']:
        prediction = classify(pattern, weight_vector, bias)
        if prediction == -1:
            correct_O += 1

    x_rate = f"{correct_X}/{len(all_dict['1'])}"
    o_rate = f"{correct_O}/{len(all_dict['-1'])}"

    percent_x = correct_X / len(all_dict['1']) * 100
    percent_o = correct_O / len(all_dict['-1']) * 100

    print(f"Correctly Classified {dict_name} 'X': {x_rate}", f"With {round(percent_x, 2)}%")
    print(f"Correctly Classified {dict_name} 'O': {o_rate}", f"With {round(percent_o, 2)}%")


if __name__ == '__main__':
    with open(learning_output, 'r', encoding='utf-8') as learn_out:
        weight_vector = learn_out.readline(-1)
        bias = learn_out.readline(-1)

    weight_vector = ast.literal_eval(weight_vector)
    bias = int(bias)

    # checking learned data identifying
    test_dict = load_learn_data()
    test_heb_network(test_dict, 'learned')

    # checking corrupted data identifying
    corrupt_dict = load_learn_data(corrupt_data)
    test_heb_network(corrupt_dict, 'corrupted')

    # checking lost data identifying
    lost_dict = load_learn_data(lost_data)
    test_heb_network(lost_dict, 'lost')
