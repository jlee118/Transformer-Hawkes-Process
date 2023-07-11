import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
class MetaUniEventData(torch.utils.data.Dataset):
    def __init__(self, tasks, task_labels, batch_size, k_shot):
        """
        Inputs:
            inter_event_times - Array of Arrays of Numpy Arrays containing varying-length inter-event times
            batch_size - Integer denoting number of tasks in a batch (non repeating?)
            task_labels - Array of integers denoting the task of each sample
        """
        self.inputs = np.array([[x[:-1] for x in task] for task in tasks], dtype=object)
        self.task_labels = np.array(task_labels)
        self.targets = np.array([[x[1:] for x in task] for task in tasks], dtype=object)

        # this might be wrong
        self.batch_size = batch_size
        self.k_shot = k_shot

    def __len__(self):
        # this might be wrong, needs to refer to length of data in a task probably
        return self.batch_size

    def create_batches(self):
        """

        """
        self.support_batch = []
        self.query_batch = []
        for b in range(self.batch_size):  # for each batch
            sampled_task = np.random.choice(np.unique(self.task_labels))
            task_inds = np.where(self.task_labels == sampled_task)
            same_dist_inputs = self.inputs[task_inds]
            same_dist_outputs = self.inputs[task_inds]

            sample_inds = np.random.choice(np.arange(len(same_dist_inputs)), self.k_shot, False)
            np.random.shuffle(sample_inds)

            support_inputs, support_outputs = pad_sequence(same_dist_inputs[sample_inds[:-1]]), pad_sequence(same_dist_outputs[:-1])
            query_inputs, query_outputs = pad_sequence(same_dist_outputs[sample_inds[-1]]), pad_sequence(same_dist_outputs[-1])


            self.support_batch.append((support_inputs, support_outputs))
            self.query_batch.append((query_inputs, query_outputs))  # append sets to current sets

    def __getitem__(self, idx):
        # This is only getting the items (sets) in the current batch

        support_inputs, support_outputs = self.support_batch[idx]
        query_inputs, query_outputs = self.query_batch[idx]

        return support_inputs, support_outputs, query_inputs, query_outputs




