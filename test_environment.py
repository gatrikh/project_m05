import sys

REQUIRED_PYTHON = "3.8.8"


def main():
    python_version = str(sys.version_info.major) + '.' + str(sys.version_info.minor) + '.' + str(sys.version_info.micro)

    if REQUIRED_PYTHON != python_version:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                REQUIRED_PYTHON, python_version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
