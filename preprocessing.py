class GenericBuffer:
    """
    A FIFO buffer to store the weights of the vehicles.
    """
    def __init__(self, size, label=None):
        self.size = size
        self.buffer = []
        self.label = label


    def add(self, item):
        """
        Add an item to the buffer in the first position.
        """
        # add item to position 0:
        self.buffer.insert(0, item)
        if len(self.buffer) > self.size:
            # too much info in the buffer, remove the last item
            self.buffer.pop()


    def get(self):
        """
        This is a FIFO buffer, so we return the last item.
        """
        if len(self.buffer) > 0:
            return self.buffer[-1]
        else:
            return None

    def pop(self):
        """
        Remove the last item from the buffer.
        """
        if len(self.buffer) > 0:
            self.buffer.pop()


    def __len__(self):
        return len(self.buffer)
