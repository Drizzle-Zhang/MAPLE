# open file in write mode


import os


def write_stringList_2File(fileName, stringList):
    with open(fileName, 'w') as fp:
        for item in stringList:
            fp.write("%s\n" % item)


def read_stringList_FromFile(fileName):
    """
    Reads a file and returns a list of strings.
    Each string is a line from the file.
    The newline character is stripped from each line.
    
    Parameters:
        fileName: The name of the file to read.
        
    Returns:
        A list of strings, each string is a line from the file.
        The newline character is stripped from each line.
    """
    result_list = []
    with open(fileName, 'r') as fp:
        for line in fp:
            result_list.append(line.strip())
    return result_list



class FileUtils(object):
    def __init__(self):
        super().__init__()
    pass

    @staticmethod
    def makedir(dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        f = open(os.path.join(dirs, filename), "a")
        f.close()
