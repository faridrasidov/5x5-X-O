import numpy as np

from learn import corrupt_data, lost_data

X_pattern = np.array([1, -1, 1, -1, 1,
                      -1, 1, -1, 1, -1,
                      1, -1, 1, -1, 1,
                      -1, 1, -1, 1, -1,
                      1, -1, 1, -1, 1])

O_pattern = np.array([-1, 1, 1, 1, -1,
                      1, -1, -1, -1, 1,
                      1, -1, -1, -1, 1,
                      1, -1, -1, -1, 1,
                      -1, 1, 1, 1, -1])


def corrupt_pattern(pattern, noise_level, lost:bool = False):
    """
    corrupts the pattern by flipping some bits based on noise_level.
    """
    if not lost:
        corrupt_val = -1
    else:
        corrupt_val = 0
    corrupted_pattern = pattern.copy()
    num_flips = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), num_flips, replace=False)

    for idx in flip_indices:
        corrupted_pattern[idx] *= corrupt_val

    return corrupted_pattern


def generate_dataset(num_samples, noise_level, is_lost: bool = False):
    """
    generates dataset by patterns.
    """
    dataset = []
    for _ in range(num_samples // 2):
        corrupted_X = corrupt_pattern(X_pattern, noise_level, is_lost)
        dataset.append(np.append(corrupted_X, 1))

        corrupted_O = corrupt_pattern(O_pattern, noise_level, is_lost)
        dataset.append(np.append(corrupted_O, -1))

    return np.array(dataset)


def save_dataset_to_file(dataset, to_where):
    """
    save the dataset to the txt file.
    """
    np.savetxt(to_where, dataset, fmt='%d', delimiter=' ')


if __name__ == '__main__':
    corrupt_noise_level = 0.1
    corrupt_num_samples = 500
    corrupt_dataset = generate_dataset(corrupt_num_samples, corrupt_noise_level)
    lost_dataset = generate_dataset(corrupt_num_samples, corrupt_noise_level, is_lost=True)
    save_dataset_to_file(corrupt_dataset, corrupt_data)
    save_dataset_to_file(lost_dataset, lost_data)
    print(f"datasets with {corrupt_num_samples} samples generated and saved.")
