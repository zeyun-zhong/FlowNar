import random, math


class DictWithTo(dict):
    def to(self, *args, **kwargs):
        return self


def rand_bool():
    return bool(random.getrandbits(1))


def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)


def get_previous_frames_before_inserting_memory(conversation: list[dict]):
    num_frames = 0
    out = []
    for conv in conversation:
        if conv['role'] == 'stream':
            num_frames += conv['num_frames']
        elif conv['role'] == 'memory':
            out.append(num_frames)
    return out
