import cv2,sys
import numpy as np
import	dlib



PREDICTOR_PATH = "/home/mihir/dlib-18.16/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

CORRECT_COLOUR_BLUR_FACTOR = 0.6

FACE_POINTS = list(range(17,18))
MOUTH_POINTS = list(range(48,61))
LEFT_BROW_POINTS = list(range(21,27))
RIGHT_BROW_POINTS = list(range(17,21))
LEFT_EYE_POINTS = list(range(42,48))
RIGHT_EYE_POINTS = list(range(36,42))
NOSE_POINTS = 	list(range(27,35))
JAW_POINTS = list(range(0,17))

ALIGN_POINTS = (LEFT_BROW_POINTS+LEFT_EYE_POINTS+RIGHT_EYE_POINTS+RIGHT_BROW_POINTS+MOUTH_POINTS+NOSE_POINTS)

OVERLAY_POINTS = [
	LEFT_BROW_POINTS+LEFT_EYE_POINTS+RIGHT_EYE_POINTS+RIGHT_BROW_POINTS+MOUTH_POINTS+NOSE_POINTS	
]

detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)




class TooManyFaces(Exception):
 	pass

class NoFaces(Exception):
 	pass


def read_im_and_landmarks(fname):
	im = cv2.imread(fname, cv2.IMREAD_COLOR)
	im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
	s = get_landmarks(im)
	return im, s


def get_landmarks(im):
 	rects = detector(im,1)

 	if len(rects) > 1:
 		raise TooManyFaces
 	if len(rects) == 0:
 		raise NoFaces

 	return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])


def get_images_landmarks(path):
	im = cv2.imread(path,cv2.IMREAD_COLOR)
	im = cv2.resize(im,(im.shape[1]*SCALE_FACTOR,im.shape[0]*SCALE_FACTOR))

	s = get_landmarks(im)

	return im,s



def transformation_points(points1,points2):
	points1 = points1.astype(np.float64)
	points2	= points2.astype(np.float64)

	c1 = np.mean(points1, axis=0)
	c2 = np.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2

	s1 = np.std(points1)
	s2 = np.std(points2)
	points1 /= s1
	points2 /= s2

	U, S, Vt = np.linalg.svd(points1.T*points2)
	R = (U*Vt).T

	return np.vstack([np.hstack(( (s2/s1)*R,c2.T - (s2/s1)*R*c1.T)), np.matrix([0. ,0. ,1.])])

def warp_im(im, M, dshape):
	output_im = np.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im, M[:2],(dshape[1],dshape[0]),dst=output_im,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)

	return output_im

def correct_colours(im1, im2, landmarks1):
	blur_ammount = CORRECT_COLOUR_BLUR_FACTOR * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS],axis = 0)-np.mean(landmarks1[RIGHT_EYE_POINTS],axis = 0))

	blur_ammount = int(blur_ammount)
	if blur_ammount % 2 == 0:
		blur_ammount += 1
	im1_blur = cv2.GaussianBlur(im1,(blur_ammount,blur_ammount),0)
	im2_blur = cv2.GaussianBlur(im2,(blur_ammount,blur_ammount),0)

	im2_blur += (128 * (im2 <= 1.0)).astype(im2_blur.dtype)

	return (im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64))

def draw_convex_hull(im, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
	im = np.zeros(im.shape[:2], dtype=np.float64)
	for group in OVERLAY_POINTS:
		draw_convex_hull(im,landmarks[group], color=1)

	im = np.array([im, im, im]).transpose((1, 2, 0))

	im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
	im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
	return im

def faceSwap(im1, im2):
	landmarks1 = get_landmarks(im1)
	im2, landmarks2 = read_im_and_landmarks(im2)

	M = transformation_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
	
	mask = get_face_mask(im2, landmarks2)
	warped_mask = warp_im(mask, M, im1.shape)
	combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

	warped_im2 = warp_im(im2, M, im1.shape)
	warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

	output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

	cv2.imwrite('output.jpg', output_im)
	image = cv2.imread('output.jpg')

	
	return image


cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation = cv2.INTER_LINEAR)
	frame = cv2.flip(frame, 1)
	output = faceSwap(frame, sys.argv[1])
	frame = cv2.resize(frame, None, fx=1.75, fy=1.5, interpolation = cv2.INTER_LINEAR)
	cv2.imshow("blabla",output)
	if cv2.waitKey(5) & 0xFF == 27:
		break

cap.release()
cv2.destroyALLwWindows()































