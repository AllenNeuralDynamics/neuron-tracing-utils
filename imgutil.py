from jpype import JException


# FIXME this would probably be a lot faster within Java
def local_intensities(imglib_iterable):
    cursor = imglib_iterable.cursor()
    while cursor.hasNext():
        cursor.fwd()
        try:
            yield cursor.get().get()
        except JException as e:
            # ArrayIndexOutOfBounds is expected here.
            # cursors don't check whether they are in a valid
            # portion of the image when accessing pixels.
            continue
