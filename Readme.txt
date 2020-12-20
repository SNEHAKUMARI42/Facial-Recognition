step1:

included these libraries

1:-imutils
2:-dlib
3:-cv2
4:-numpy
5:-math
6:-os

download this file "shape_predictor_68_face_landmarks.dat" from internet and place
this file here in this same folder named "Attempt7"

step2:

after adding all the libraries

open Capture_Photo.py at line number 49 you will see
this line  "cv2.imwrite(('Faces/Shivam.jpg'), frame)"
replace "Shivam.jpg" with "YourName.jpg"

then run this file

you will see 2 window .

In first window you will see your simple video and in 2nd window you will see
your face with many dots

press "c" to capture image , press "c" only when all the dots are on border of your face,
boder of your eyes, border of your lips.

check in folder named faces for your image.

if you get your image , congracts you successfully saved your image

if not repeat step 2 again



Step3:

run Attempt7.py

you will see your face get detected