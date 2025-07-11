# open file in write mode


import os


def write_stringList_2File(fileName, stringList):
    """
    Write a list of strings to a file, one string per line.
    
    Args:
        fileName (str): Path to the output file
        stringList (list): List of strings to write
    """
    with open(fileName, 'w') as fp:
        for item in stringList:
            fp.write("%s\n" % item)


def read_stringList_FromFile(fileName):
    """
    Read a file and return a list of strings, one per line.
    
    Each string is a line from the file with the newline character stripped.
    
    Args:
        fileName (str): Path to the file to read
        
    Returns:
        list: List of strings, each string is a line from the file
    """
    result_list = []
    with open(fileName, 'r') as fp:
        for line in fp:
            result_list.append(line.strip())
    return result_list



class FileUtils:
    """
    Utility class for file and directory operations.
    
    This class provides static methods for common file operations
    used throughout the MAPLE pipeline.
    """
    
    def __init__(self):
        super().__init__()
    pass

    @staticmethod
    def makedir(dirs):
        """
        Create a directory if it doesn't exist.
        
        Args:
            dirs (str): Path to the directory to create
        """
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        """
        Create an empty file in the specified directory.
        
        Args:
            dirs (str): Directory where the file should be created
            filename (str): Name of the file to create
        """
        f = open(os.path.join(dirs, filename), "a")
        f.close()
