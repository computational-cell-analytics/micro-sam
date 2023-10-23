

# See
# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


# we probably want this to be a data class
# and I am not sure which singleton pattern to go with
@singleton
class SamState:
    def __init__(self):
        self.image_embeddings = None
