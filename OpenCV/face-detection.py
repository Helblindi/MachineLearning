import cv2

# load classifiers
face_cascade = cv2.CascadeClassifier('Data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Data/haarcascades/haarcascade_eye.xml')

# load image and detect faces from gray scale image
img = cv2.imread('Data/Faces/da_silva_family.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# resize image
# img = cv2.resize(img, dsize=None, fx=.25, fy=.25)

# display results
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
