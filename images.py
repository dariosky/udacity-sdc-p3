# Working with the training images
from pathlib import Path
from sklearn.utils import shuffle

CORRECTION_LEVEL = 0.15  # intensitiy of correction for the left/right views (move to the center)
STEERING_MAX = 0.6  # cap the steer to this amount


class ImageSet:
    def __init__(self, name: str, correction: float = 0):
        self.name = name
        self.correction = correction
        self.images = []
        self.steers = []
        self.max_steer = None
        self.min_steer = None

    def append(self, imgpath, steer):
        if imgpath.exists():
            self.images.append(imgpath)
            steer += self.correction
            if steer > STEERING_MAX:
                steer = STEERING_MAX
            elif steer < -STEERING_MAX:
                steer = -STEERING_MAX
            self.steers.append(steer)
            if self.max_steer is None or steer > self.max_steer:
                self.max_steer = steer
            if self.min_steer is None or steer < self.min_steer:
                self.min_steer = steer
            return True

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
        return new

    def __str__(self, *args, **kwargs):
        return "{name} - {size} images in the set.".format(
            size=len(self.images),
            name=self.name
        )

    def __getitem__(self, item):
        """ This support custom cuts and slices on the images sets """
        r = ImageSet(name=self.name + " extract", correction=self.correction)
        r.images = self.images[item]
        r.steers = self.steers[item]
        return r

    def shuffle(self):
        """ Shuffle the sets """
        self.images, self.steers = shuffle(self.images, self.steers)


def set_from_folder(folder: Path, name="friendly name", nozeroes=False):
    """ Return a Training set reading csv and images from folder
    :type nozeroes: bool when true the zero steering are filtered out
    """
    print("Parsing %s" % folder, end=" ")
    csv_path = folder / 'driving_log.csv'
    if not csv_path.exists():
        raise Exception("Invalid folder %s" % folder)
    import csv

    imgset = ImageSet(name)

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
            if nozeroes and steer < 0.01:
                continue
            center, left, right = [
                None if not s else
                Path(s) if Path(s).is_absolute() else working_dir / Path(s)
                for s in (center, left, right)
                ]
            valid = False
            if center:
                valid = imgset.append(center, steer)
            if valid:
                # if we don't have the center image, ignore the others
                if left:
                    imgset.append(left, steer + CORRECTION_LEVEL)
                if right:
                    imgset.append(right, steer - CORRECTION_LEVEL)
    imgset.shuffle()
    print(len(imgset.images))
    return imgset


print("Balancing leftright of %s" % CORRECTION_LEVEL)


def get_sample_set(nozeroes=False):
    training_folder = Path("/home/dario/tmp/driverecords/data")
    return set_from_folder(training_folder, "Udacity sample",
                           nozeroes=nozeroes)


def get_training_set(nozeroes=True):
    training_folder = Path("/home/dario/tmp/driverecords")
    return set_from_folder(training_folder, "My records",
                           nozeroes=nozeroes)


def get_refinement_set_1(nozeroes=True):
    return set_from_folder(Path("/home/dario/tmp/driverecords/30 morning session"),
                           "Refinement 1",
                           nozeroes=nozeroes)


def get_refinement_set_2(nozeroes=True):
    return set_from_folder(Path("/home/dario/tmp/driverecords/full_screen"),
                           "Refinement 2",
                           nozeroes=nozeroes)


if __name__ == '__main__':
    print(get_sample_set() +
          get_training_set() +
          get_refinement_set_1() +
          get_refinement_set_2())
