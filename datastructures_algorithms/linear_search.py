def linear_search(list, target):
    """
    Returns the index position of the index if found, else return None
    """
    for i in range(0, len(list)):
        if list[i] == target:
            return i
    return None

def verify(index):
    if index is not None:
        print("Target found at index: ", index)
    else:
        print('Target not found')

numbers = [1,2,3,4,5,6,7,8,9]
result = linear_search(numbers, 6)
verify(result)