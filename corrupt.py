import numpy as np

from learn import corrupt_data

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


def corrupt_pattern(pattern, noise_level):
    """
    corrupts the pattern by flipping some bits based on noise_level.
    """
    corrupted_pattern = pattern.copy()
    num_flips = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), num_flips, replace=False)

    for idx in flip_indices:
        corrupted_pattern[idx] *= -1

    return corrupted_pattern


def generate_dataset(num_samples, noise_level):
    """
    generates dataset by patterns.
    """
    dataset = []
    for _ in range(num_samples // 2):
        corrupted_X = corrupt_pattern(X_pattern, noise_level)
        dataset.append(np.append(corrupted_X, 1))

        corrupted_O = corrupt_pattern(O_pattern, noise_level)
        dataset.append(np.append(corrupted_O, -1))

    return np.array(dataset)


def save_dataset_to_file(dataset):
    """
    save the dataset to the txt file.
    """
    np.savetxt(corrupt_data, dataset, fmt='%d', delimiter=' ')


if __name__ == '__main__':
    corrupt_noise_level = 0.1
    corrupt_num_samples = 100
    corrupt_dataset = generate_dataset(corrupt_num_samples, corrupt_noise_level)

    save_dataset_to_file(corrupt_dataset)
    print(f"dataset with {corrupt_num_samples} samples generated and saved.")
