import cv2
import numpy as np

def simple_seemless_cloning():
    # Read images
    src = cv2.imread("Data/Other images/airplane.jpg")
    src = cv2.resize(src, dsize=None, fx=.25, fy=.25)
    dst = cv2.imread("Data/Other images/sky.jpg")

    # Create a rough mask around the airplane.
    src_mask = np.zeros(src.shape, src.dtype)
    poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))

    # This is where the CENTER of the airplane will be placed
    center = (800, 100)

    # Clone seamlessly.
    output = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)
    cv2.imshow("image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    cv2.imwrite("Data/Other images/opencv-seamless-cloning-example.jpg", output);


def complex_seemless_cloning():
    return 0


def main():
    simple_seemless_cloning()


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()
