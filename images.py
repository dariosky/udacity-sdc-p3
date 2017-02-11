# Working with the training images
import csv
from collections import deque, defaultdict
from pathlib import Path
from statistics import mean

from sklearn.utils import shuffle

CORRECTION_LEVEL = 0.15  # intensitiy of correction for the left/right views (move to the center)
STEERING_MAX = 0.6  # cap the steer to this amount


class ImageSet:
    STRAIGHT_TRESHOLD = 0.1  # we consider straight when abs(steer) < treshold

    def __init__(self, name: str = "Unnamed dataset",
                 correction: float = 0,
                 skip_zeros=False,  # the ~zero steering are filtered out
                 skip_sides_when_zero=True,  # the left/right are considered only for curves
                 ):
        self.name = name
        self.correction = correction
        self.images = []
        self.steers = []
        self.max_steer = None
        self.min_steer = None

        self.skip_zeros = skip_zeros
        self.skip_sides_when_zero = skip_sides_when_zero

        # the named_groups are things like 'center', 'left' and 'right'
        self.counters = defaultdict(int)  # the counters, one per named_group
        # we'll keep a rolling averages of steers per named_group
        self.rolling_windows = defaultdict(lambda: deque(maxlen=3))

        self.skipped_counters = defaultdict(int)

    def append(self, imgpath, steer, named_group='center', real_steer=None):
        if imgpath.exists():
            abs_steer = abs(steer)
            if self.skip_zeros and abs_steer < self.STRAIGHT_TRESHOLD:
                # skip straight
                self.rolling_windows[named_group].clear()
                self.skipped_counters['zeros'] += 1
                return False

            if (self.skip_sides_when_zero
                and real_steer is not None
                and abs(real_steer) < self.STRAIGHT_TRESHOLD
                and named_group != 'center'):
                # skip sides when straight
                self.rolling_windows[named_group].clear()
                self.skipped_counters['side on zero'] += 1
                return False

            steer += self.correction

            if steer > STEERING_MAX:  # cap the steering
                steer = STEERING_MAX
            elif steer < -STEERING_MAX:
                steer = -STEERING_MAX

            self.counters[named_group] += 1
            self.rolling_windows[named_group].append(
                steer)  # add the steer to the rolling window
            steer = mean(self.rolling_windows[named_group])

            self.images.append(imgpath)
            self.steers.append(steer)

            # keep the max/min steers
            if self.max_steer is None or steer > self.max_steer:
                self.max_steer = steer
            if self.min_steer is None or steer < self.min_steer:
                self.min_steer = steer
            return True
        else:
            # image does not exists
            # print("Not found %s" % imgpath)
            self.skipped_counters['%s image not found' % named_group] += 1

    def __len__(self):
        return len(self.images)

    def __add__(self, other):
        if other is None:
            return self
        assert isinstance(other, ImageSet), "Add requeires to ImageSet"
        new = ImageSet(name=self.name + "+" + other.name)
        new.max_steer = max(self.max_steer, other.max_steer)
        new.min_steer = max(self.min_steer, other.min_steer)
        new.images = self.images + other.images
        new.steers = self.steers + other.steers

        for counter_name in ('counters', 'skipped_counters'):  # sum the counters
            counter_a, counter_b = getattr(self, counter_name), getattr(other, counter_name)
            new_counter = counter_a.copy()
            for k, v in counter_b.items():
                new_counter[k] += v
            setattr(new, counter_name, new_counter)
        return new

    def __str__(self, *args, **kwargs):
        desc = "{name} - {size} images in the set".format(
            size=len(self.images),
            name=self.name
        )
        if self.skipped_counters:
            desc += "- Skipped: [" + \
                    ", ".join(
                        ["%s: %d" % (name, val) for name, val in self.skipped_counters.items()]
                    ) + "]"
        if self.counters:
            desc += " - " + ", ".join(
                ["%s: %d" % (name, val) for name, val in self.counters.items()]
            )
        return desc

    def __getitem__(self, item):
        """ This support custom cuts and slices on the images sets """
        r = ImageSet(name=self.name + " extract", correction=self.correction)
        r.images = self.images[item]
        r.steers = self.steers[item]
        return r

    def shuffle(self):
        """ Shuffle the sets """
        self.images, self.steers = shuffle(self.images, self.steers)


def clear_csv_path(p, working_dir):
    """ Given a path from the csv, clean it so it's absolute but relative to to the csv """
    if not p:
        return None
    if not p.startswith('IMG'):
        p = p[p.rfind("IMG"):]  # get the relative path, it starts with the last IMG
    return working_dir / Path(p)


def set_from_folder(folder: Path, **kwargs
                    ):
    """ Return a Training set reading csv and images from folder
    """
    print("Parsing %s" % folder, end=" ")
    csv_path = folder / 'driving_log.csv'
    if not csv_path.exists():
        raise Exception("Invalid folder %s" % folder)

    imgset = ImageSet(**kwargs)

    with csv_path.open() as csvfile:
        data_reader = csv.reader(csvfile)
        working_dir = csv_path.parent
        for row in data_reader:
            center, left, right, steer, throttle, break_value, speed = [x.strip() for x in row]
            try:
                steer, throttle = map(float, [steer, throttle])
            except ValueError:
                continue
            # clean the images, and set them relative to the csv if needed
            center, left, right = [clear_csv_path(x, working_dir) for x in (center, left, right)]
            valid = False
            if center:
                valid = imgset.append(center, steer)
            if valid:
                # if we don't have the center image, ignore the others
                if left:
                    imgset.append(left, steer + CORRECTION_LEVEL, 'left', real_steer=steer)
                if right:
                    imgset.append(right, steer - CORRECTION_LEVEL, 'right', real_steer=steer)
    imgset.shuffle()
    print(len(imgset.images))
    return imgset


print("Balancing leftright of %s" % CORRECTION_LEVEL)


def get_sample_set():
    training_folder = Path("/home/dario/tmp/driverecords/sample")
    return set_from_folder(training_folder, name="Udacity sample")


def get_training_set():
    training_folder = Path("/home/dario/tmp/driverecords/dev")
    return set_from_folder(training_folder, name="My records")


def get_myfirst_datasets():
    """ A dataset from all the stable recordings"""
    result = set_from_folder(Path('/home/dario/tmp/driverecords/janstable/28'),
                             name='First stable dataset')
    result += set_from_folder(Path('/home/dario/tmp/driverecords/janstable/30 morning session'),
                              name="2nd stable dataset")
    result += set_from_folder(Path('/home/dario/tmp/driverecords/janstable/full_screen'),
                              name="3rd stable dataset")
    return result


if __name__ == '__main__':
    print("These are my datasets, we'll use them on model.py for training")
    print(get_sample_set())
    print(get_myfirst_datasets())
    print(get_training_set())
