import math
import torch
import numpy as np


class BatchDataloader:
    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, *tensors, batch_size=1, mask=None):
        # Returns a tuple of arrays, one for each dimension, containing the indices
        # of the non-zero elements in that dimension.
        nonzero_idx, = np.nonzero(mask)
        self.tensors = tensors
        self.batch_size = batch_size
        self.mask = mask

        # return start_idx, end_idx of training and validation sets
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx) + 1
        else:
            self.start_idx = 0
            self.end_idx = 0

    # When defining a custom dataset class, you'd ordinarily subclass torch.utils.data.Dataset and define __len__()
    # and __getitem__().
    # However, for cases where you want sequential but not random access, you can use an
    # iterable-style dataset (https://pytorch.org/docs/stable/data.html#iterable-style-datasets).
    # To do this, you instead subclass torch.utils.data.IterableDataset and define __iter__().
    # Whatever is returned by __iter__() should be a proper iterator; it should maintain state (if necessary)
    # and define __next__() to obtain the next item in the sequence. __next__() should raise StopIteration
    # when there's nothing left to read. In your case with an infinite dataset, it never needs to do this.

    # https://juejin.cn/post/7073726994856280078
    # The meaning of iteration is similar to cycle, each repeated process is called an iterative process,
    # and the result of each iteration will be used as the initial value of the next iteration.
    # Containers that provide iteration methods are called iterators. Commonly encountered iterators include
    # sequences (lists, tuples, and strings) and dictionaries. These data structures support iterative operations.
    #
    # There are two magic methods for implementing iterators: __iter__(self)__ and __next__(self)
    #
    # If a container is an iterator, it must implement __iter__(self) the magic method, which actually returns
    # an iterator (usually the iterator itself). The next thing to focus on is the __next__(self) magic method,
    # because it determines the rules of iteration.
    #
    # In general, an iterator satisfies the following properties:
    #    + An iterator is an object
    #    + Iterators can be called by the next() function and return a value
    #    + The iterator can be called by the iter() function and returns an iterator (it can be itself)
    #    + Returns a series of values sequentially when called by next()
    #    + If the end of the iteration is reached, a StopIteration exception is thrown
    #    + The iterator can also have no end, as long as it is called by next(), it will definitely return a value
    #    + In Python, the next() built-in function calls the next() method of the object
    #    + In Python, the iter() built-in function calls the object's iter() method
    #    + An object that implements the iterator protocol can be iterated by a for statement loop until termination
    #

    # calls the __next__() method to get the iteration.
    def __next__(self):
        if self.start == self.end_idx:  # Stopping when reaching to ending of training and validation sets
            raise StopIteration

        end = min(self.start + self.batch_size, self.end_idx)   # calculating end point of each interation
        batch_mask = self.mask[self.start:end]  # taking data for each batch_size

        # if data of batch is equal zero
        while sum(batch_mask) == 0:
            self.start = end    # assign start to end
            end = min(self.start + self.batch_size, self.end_idx)   # calculating end point of each interation
            batch_mask = self.mask[self.start:end]                  # taking data for each batch_size

        # converting data each batch of data to array numpy
        batch = [np.array(t[self.start:end]) for t in self.tensors]
        self.start = end                # assign start to end after each batch
        self.sum += sum(batch_mask)     # data accumulation of batches

        # convert array numpy of each tensor on a batch to tensor format
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch]

    # Defines the behavior when iterating over the elements in the container
    def __iter__(self):
        # initialization start index and sum value for first interation
        self.start = self.start_idx
        self.sum = 0
        return self

    # Defines the behavior when called by a function, generally returns the number of elements in the iterator
    def __len__(self):
        # initialization count and start index
        count = 0
        start = self.start_idx

        # if not reaching to ending of training and validation sets
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)    # calculating end point of each interation
            batch_mask = self.mask[start:end]                   # taking data for each batch_size

            # # if data of batch is different zero
            if sum(batch_mask) != 0:
                count += 1  # increasing count value
            start = end     # assign start to end of each batch
        return count
