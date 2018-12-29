import time
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def GetTimeFromMs(ms):
    ms = int(ms)
    seconds = (ms/1000) % 60
    seconds = int(seconds)
    minutes = (ms/(1000*60)) % 60
    minutes = int(minutes)
    hours = (ms / (1000*60*60)) % 24
    return "%02d:%02d:%02d.%f" % (hours, minutes, seconds, ms % 1000)


def PlotMatch(max_loc, w, h, haystack, needle, method):
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(haystack, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(needle, cmap='gray')
    plt.title('Template to find'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(haystack, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    plt.show()

def ImageRescale(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class TemplateMatcher:

    # 1. haystack and needle parameters should be pixel arrays
    # 2. haystack and needle should be in cv2.IMREAD_COLOR format
    # 3. haystack is bigger than needle in both dimensions

    def __init__(self, needle, threshold):
        self.needle = needle
        self.threshold = threshold

    def match(self, haystack) -> bool:
        haystack = haystack.copy()
        # cv2.TM_CCOEFF cv2.TM_CCOEFF_NORMED cv2.TM_CCORR cv2.TM_CCORR_NORMED cv2.TM_SQDIFF cv2.TM_SQDIFF_NORMED

        h, w = self.needle.shape[:2]
        matches = cv2.matchTemplate(haystack, self.needle, cv2.TM_CCOEFF_NORMED)
        loc = np.where(matches >= self.threshold)
        loc = list(zip(*loc[::-1]))

        #for point in loc:
        #    cv2.rectangle(haystack, point, (point[0] + w, point[1] + h), (0, 0, 255), 2)
        #    plotMatch(point, w, h, haystack, self.needle, cv2.TM_CCOEFF_NORMED)

        return len(loc) > 0


class MultiScaleTemplateMatcher(TemplateMatcher):

    def __init__(self, needle, threshold):
        TemplateMatcher.__init__(self, None, threshold)
        self.needles = [ImageRescale(needle, newWidth) for newWidth in range(24, 160, 10)]

    def match(self, haystack) -> bool:
        for needle in self.needles:
            self.needle = needle
            isMatch = super(MultiScaleTemplateMatcher, self).match(haystack)
            if isMatch:
                return True
        return False


class VideoFrameGrabber:

    def __init__(self, videoPath, rescaleWidth, seconds):
        self.video = cv2.VideoCapture(videoPath)
        self.frameWidth = rescaleWidth
        self.seconds = seconds
        if not self.video.isOpened():
            raise Exception("Could not open video!")

    def iterate(self):
        try:
            actualFps = int(self.video.get(cv2.CAP_PROP_FPS))
            multiplier = actualFps * self.seconds
            while self.video.isOpened():
                frameId = int(round(self.video.get(1)))
                isFrameGrabbed, frame = self.video.read()
                if isFrameGrabbed:
                    if frameId % multiplier == 0:
                        timeMs = int(self.video.get(cv2.CAP_PROP_POS_MSEC))
                        #frame = ImageRescale(frame, self.frameWidth)
                        yield timeMs, frame
                else:
                    break
        finally:
            self.video.release()

def main():
    args = sys.argv
    pikachu = cv2.imread('pikachu_full.jpg', cv2.IMREAD_COLOR)
    matcher = MultiScaleTemplateMatcher(pikachu, 0.90)
    #fullImage = cv2.imread('pikachu_haystack.jpg', cv2.IMREAD_COLOR)
    #print(matcher.match(fullImage))
    videoGrabber = VideoFrameGrabber('p_s01e10.mp4', 320, 1)

    start = time.time()
    for timeMs, frame in videoGrabber.iterate():
        if matcher.match(frame):
            print('Found at: ' + GetTimeFromMs(timeMs))
            break
        else:
            # print(str(timeMs) + 'ms')
            pass
    print('Elapsed time: ' + str(int(time.time() - start) + 1) + " seconds")

main()
